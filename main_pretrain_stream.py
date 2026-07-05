import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import re
import csv
import json
import argparse
from datetime import datetime

from tokenizer import Tokenizer
from datasets import load_dataset

from transformer import Decoder

# ── System ────────────────────────────────────────────────────────────────────
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_count   = os.cpu_count() or 1
# Keep this at zero for the simple stateful streaming implementation below.
# Multiple workers require per-worker dataset-state checkpointing.
num_workers = 0
TOKENIZER_WORKERS = max(1, cpu_count - 1)
ENCODE_WORKERS    = max(1, cpu_count - 1)  # retained for CLI compatibility; unused

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "./data/fineweb-edu"
NUM_DOCS       = 9_672_101  # Max documents in FineWeb-EDU sample-10BT.
VOCAB_SIZE     = 32768
SPECIAL_TOKENS = ["<|endoftext|>"]
TOKEN_BUDGET   = 7_000_000_000 # 0 = disabled (epoch mode), >0 = token-budget mode
VAL_TOKENS     = 262_144    # 8 val batches at batch_size=32, context_length=1024.
CHECKPOINT_INTERVAL_TOKENS = 1_000_000_000
CHECKPOINT_DIR    = "./checkpoints/pretrain"
CHECKPOINT_PREFIX = "qwen25_coder_05b"

# ── Model ─────────────────────────────────────────────────────────────────────
ARCHITECTURE_REFERENCE = "Qwen2.5-Coder-0.5B"
context_length = 1024       # maximum sequence length
d_model        = 896        # embedding dimension
swiglu_d       = 4864       # SwiGLU hidden dimension
num_heads      = 14         # number of attention heads
num_key_value_heads = 2     # number of grouped-query attention KV heads
num_layers     = 24         # number of transformer layers
rope_base      = 1_000_000.0

# ── Training ──────────────────────────────────────────────────────────────────
batch_size     = 4
learning_rate  = 3e-4
eval_interval  = 1000    # log every N steps
training_dtype = torch.bfloat16
use_mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()


def bf16_autocast():
    return torch.autocast(
        device_type=device.type,
        dtype=training_dtype,
        enabled=use_mixed_precision,
    )


@torch.no_grad()
def compute_perplexity(decoderLMmodel, data_loader):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        with bf16_autocast():
            loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.1):
    """Autoregressive sampling from the decoder."""
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to block_size if the sequence gets too long
        x_cond = x[:, -context_length:]
        with bf16_autocast():
            logits = model(x_cond)                    # no targets → returns logits
        logits = logits[:, -1, :] / temperature       # last position only
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    model.train()
    return tokenizer.decode(x.squeeze(0).tolist())

def load_data(num_docs=NUM_DOCS):
    """Return a lazy FineWeb-EDU stream; dataset shards are not cached locally."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    if num_docs is not None and num_docs > 0:
        ds = ds.take(num_docs)
    return ds


class TokenBlockDataset(Dataset):
    """Small map-style dataset used only for the in-memory validation tokens."""

    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return max(0, (len(self.token_ids) - 1) // self.block_size)

    def __getitem__(self, index):
        start = index * self.block_size
        chunk = self.token_ids[start:start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class StreamingLMDataset(IterableDataset):
    """Tokenize streamed documents and emit contiguous next-token blocks.

    With num_workers=0, state_dict() captures both the Hugging Face stream
    position and the unconsumed token buffer, so checkpoint resume does not
    need to replay the whole dataset.
    """

    def __init__(self, source, tokenizer_path, eos_id, block_size, skip_blocks=0):
        super().__init__()
        self.source = source
        self.tokenizer_path = tokenizer_path
        self.eos_id = eos_id
        self.block_size = block_size
        self.skip_blocks = skip_blocks
        self.token_buffer = []
        self.buffer_start = 0
        self.blocks_emitted = 0

    def __iter__(self):
        tokenizer = Tokenizer.load(self.tokenizer_path)

        for example in self.source:
            ids = tokenizer.encode(example["text"])
            self.token_buffer.extend(ids)
            self.token_buffer.append(self.eos_id)

            while len(self.token_buffer) - self.buffer_start >= self.block_size + 1:
                start = self.buffer_start
                end = start + self.block_size + 1
                chunk = self.token_buffer[start:end]

                # Advance state before yielding. With num_workers=0, a checkpoint
                # taken after a batch then points exactly after that batch.
                self.buffer_start += self.block_size
                self.blocks_emitted += 1

                if self.buffer_start >= 65_536:
                    self.token_buffer = self.token_buffer[self.buffer_start:]
                    self.buffer_start = 0

                if self.skip_blocks > 0:
                    self.skip_blocks -= 1
                    continue

                block = torch.tensor(chunk, dtype=torch.long)
                yield block[:-1], block[1:]

    def state_dict(self):
        source_state = self.source.state_dict() if hasattr(self.source, "state_dict") else None
        return {
            "source_state": source_state,
            "token_buffer": self.token_buffer[self.buffer_start:],
            "blocks_emitted": self.blocks_emitted,
            "skip_blocks": self.skip_blocks,
        }

    def load_state_dict(self, state_dict):
        source_state = state_dict.get("source_state")
        if source_state is not None and hasattr(self.source, "load_state_dict"):
            self.source.load_state_dict(source_state)
            self.token_buffer = list(state_dict.get("token_buffer", []))
            self.buffer_start = 0
            self.blocks_emitted = int(state_dict.get("blocks_emitted", 0))
            self.skip_blocks = int(state_dict.get("skip_blocks", 0))
        else:
            # Compatibility fallback for older `datasets` versions: replay the
            # stream and skip already-consumed blocks instead of restoring a
            # shard/example cursor.
            self.token_buffer = []
            self.buffer_start = 0
            self.blocks_emitted = 0
            self.skip_blocks = int(state_dict.get("blocks_emitted", 0))


def build_validation_dataset(tokenizer, eos_id):
    """Read only enough streamed documents to hold VAL_TOKENS in RAM."""
    token_ids = []
    docs_used = 0

    for example in load_data(NUM_DOCS):
        token_ids.extend(tokenizer.encode(example["text"]))
        token_ids.append(eos_id)
        docs_used += 1
        if len(token_ids) >= VAL_TOKENS + 1:
            break

    num_blocks = min(
        VAL_TOKENS // context_length,
        max(0, (len(token_ids) - 1) // context_length),
    )
    if num_blocks == 0:
        raise RuntimeError("Not enough streamed text to construct one validation block")

    keep_tokens = num_blocks * context_length + 1
    val_tensor = torch.tensor(token_ids[:keep_tokens], dtype=torch.long)
    return TokenBlockDataset(val_tensor, context_length), docs_used, keep_tokens - 1


def checkpoint_path(checkpoint_index):
    return os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{checkpoint_index}b.pt")

def find_latest_checkpoint():
    if not os.path.exists(CHECKPOINT_DIR):
        return None

    pattern = re.compile(rf"^{re.escape(CHECKPOINT_PREFIX)}_(\d+)b\.pt$")
    candidates = []
    for filename in os.listdir(CHECKPOINT_DIR):
        match = pattern.match(filename)
        if match:
            candidates.append((int(match.group(1)), os.path.join(CHECKPOINT_DIR, filename)))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])

def save_training_checkpoint(model, optimizer, scheduler, train_dataset,
                             checkpoint_index, step, tokens_seen, best_val_ppl,
                             no_improve, total_steps_est, total_params):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(checkpoint_index)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "data_state_dict": train_dataset.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "checkpoint_index": checkpoint_index,
        "best_val_ppl": best_val_ppl,
        "no_improve": no_improve,
        "total_steps": total_steps_est,
        "total_params": total_params,
        "model_config": {
            "architecture_reference": ARCHITECTURE_REFERENCE,
            "context_length": context_length,
            "d_model": d_model,
            "swiglu_d": swiglu_d,
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "num_layers": num_layers,
            "rope_base": rope_base,
            "vocab_size": VOCAB_SIZE,
        },
        "training_config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_dtype": str(training_dtype).replace("torch.", ""),
            "mixed_precision": use_mixed_precision,
            "token_budget": TOKEN_BUDGET,
            "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
        },
    }, path)
    print(f"Checkpoint saved -> {path} ({tokens_seen:,} tokens)")
    return path

def main():

    # ── Timestamp & output dir ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join("output", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output → {run_dir}")

    print("Loading Tokenizer")
    tokenizer_path = f"tokenizer/tokenizer_{VOCAB_SIZE}.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Streaming pretraining expects an existing tokenizer at {tokenizer_path}. "
            "Train/save the tokenizer once before launching this script."
        )

    tokenizer = Tokenizer.load(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")
    eos_id = tokenizer.bytes_to_id[b"<|endoftext|>"]

    print(f"Streaming FineWeb-EDU (up to {NUM_DOCS:,} documents)")
    print(f"Building an in-memory validation set of {VAL_TOKENS:,} tokens ...")
    val_dataset, val_docs_used, validation_tokens = build_validation_dataset(tokenizer, eos_id)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    print(
        f"Validation: {len(val_dataset):,} samples / {validation_tokens:,} tokens "
        f"from {val_docs_used:,} streamed documents"
    )

    # Loading Model
    model = Decoder(vocab_size=tokenizer.vocab_size,
                    d_model=d_model,
                    n_head=num_heads,
                    n_kv_head=num_key_value_heads,
                    swiglu_d=swiglu_d,
                    n_layer=num_layers,
                    rope_base=rope_base)
    
    print("Model Summary:")
    print(f"  Layers: {num_layers} | Q Heads: {num_heads} | KV Heads: {num_key_value_heads} | Context: {context_length}")
    print(f"  d_model: {d_model} | swiglu_d: {swiglu_d} | RoPE base: {rope_base:g}")
    print(f"  Training dtype: {str(training_dtype).replace('torch.', '')} | Mixed precision: {use_mixed_precision}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # A stream has no cheap len(), so schedule training by token budget.
    if TOKEN_BUDGET <= 0:
        raise ValueError("Streaming mode requires TOKEN_BUDGET > 0")
    tokens_per_step = batch_size * context_length
    total_steps_est = (TOKEN_BUDGET + tokens_per_step - 1) // tokens_per_step
    
    # Calculate warmup steps (e.g., 2% of total steps)
    warmup_steps = max(1, int(0.02 * total_steps_est))
    
    # 1. Warmup: linearly scale LR from near-zero to full learning_rate
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
    
    # 2. Decay: Cosine annealing for the remaining steps down to 10% of max LR
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps_est - warmup_steps, eta_min=learning_rate * 0.1)
    
    # 3. Chain them together
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_steps]
    )

    step = 0
    best_val_ppl = float("inf")
    no_improve = 0
    tokens_seen = 0
    resume_checkpoint = find_latest_checkpoint()
    resume_checkpoint_path = None
    resume_data_state = None

    if resume_checkpoint is None:
        print("No pretrain checkpoint found; training from scratch")
    else:
        checkpoint_index, resume_checkpoint_path = resume_checkpoint
        print(f"Loading checkpoint {checkpoint_index}b from {resume_checkpoint_path} ...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", step * tokens_per_step))
        best_val_ppl = checkpoint.get("best_val_ppl", best_val_ppl)
        no_improve = int(checkpoint.get("no_improve", 0))
        resume_data_state = checkpoint.get("data_state_dict")
        print(f"Resuming at step {step:,}, tokens_seen={tokens_seen:,}")

    # Reserve the first streamed documents for validation, then train on the rest.
    train_source = load_data(NUM_DOCS).skip(val_docs_used)

    # New checkpoints restore the exact stream position and token buffer. Older
    # checkpoints fall back to replaying/skipping blocks from the beginning.
    fallback_skip_blocks = 0 if resume_data_state is not None else tokens_seen // context_length
    train_dataset = StreamingLMDataset(
        source=train_source,
        tokenizer_path=tokenizer_path,
        eos_id=eos_id,
        block_size=context_length,
        skip_blocks=fallback_skip_blocks,
    )
    if resume_data_state is not None:
        train_dataset.load_state_dict(resume_data_state)
        print("Restored streaming dataset position from checkpoint")
    elif fallback_skip_blocks > 0:
        print(
            f"Warning: old checkpoint has no data state; replaying and skipping "
            f"{fallback_skip_blocks:,} blocks"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    next_checkpoint_index = tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1

    # ── Training log CSV ──────────────────────────────────────────────────────
    log_path = os.path.join(run_dir, f"training_log_{timestamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_steps", "loss", "train_ppl", "val_ppl", "lr"])

    print("Training ...")
    model.train()
    train_ppl = None

    if TOKEN_BUDGET > 0 and tokens_seen >= TOKEN_BUDGET:
        print(f"Token budget already reached by checkpoint ({tokens_seen:,} tokens)")
    else:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            with bf16_autocast():
                loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            tokens_seen += xb.numel()
            train_ppl = torch.exp(torch.tensor(loss.item())).item()

            if step % eval_interval == 0:
                val_ppl   = compute_perplexity(model, val_loader)
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Step {step}/{total_steps_est} | "
                    f"Tokens: {tokens_seen:,} | "
                    f"LR: {current_lr:.2e} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Train PPL: {train_ppl:.2f} | "
                    f"Val PPL: {val_ppl:.2f}"
                )

                # Log to CSV
                log_writer.writerow([step, total_steps_est, f"{loss.item():.6f}",
                                     f"{train_ppl:.4f}", f"{val_ppl:.4f}", f"{current_lr:.6e}"])
                log_file.flush()

                # Early Stop configs below
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    no_improve = 0
                else:
                    no_improve += 1

            while next_checkpoint_index * CHECKPOINT_INTERVAL_TOKENS <= tokens_seen:
                save_training_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_dataset=train_dataset,
                    checkpoint_index=next_checkpoint_index,
                    step=step,
                    tokens_seen=tokens_seen,
                    best_val_ppl=best_val_ppl,
                    no_improve=no_improve,
                    total_steps_est=total_steps_est,
                    total_params=total_params,
                )
                next_checkpoint_index += 1

            if TOKEN_BUDGET > 0 and tokens_seen >= TOKEN_BUDGET:
                print(f"Token budget reached at step {step} ({tokens_seen:,} tokens)")
                break
        else:
            if TOKEN_BUDGET > 0 and tokens_seen < TOKEN_BUDGET:
                print(f"Training data exhausted at {tokens_seen:,} tokens before token budget {TOKEN_BUDGET:,}")

    log_file.close()

    # Final eval
    val_ppl   = compute_perplexity(model, val_loader)
    if train_ppl is None:
        print(f"Final — Val PPL: {val_ppl:.2f}")
    else:
        print(f"Final — Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(run_dir, f"model_{timestamp}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "data_state_dict": train_dataset.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_ppl": best_val_ppl,
        "no_improve": no_improve,
    }, model_path)
    print(f"Model saved → {model_path}")

    # Generation
    prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Education is important because",
    ]
    print("\n--- Generation ---")
    generation_outputs = []
    for prompt in prompts:
        output = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")
        generation_outputs.append({"prompt": prompt, "output": output})

    # ── Save run config ───────────────────────────────────────────────────────
    run_config = {
        "timestamp": timestamp,
        "device": str(device),

        "data": {
            "streaming": True,
            "dataset": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "num_docs_limit": NUM_DOCS,
            "validation_docs": val_docs_used,
            "validation_tokens": validation_tokens,
            "val_samples": len(val_dataset),
            "stream_workers": 0,
        },

        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "tokenizer_workers": TOKENIZER_WORKERS,
            "tokenizer_path": tokenizer_path,
        },

        "model": {
            "architecture_reference": ARCHITECTURE_REFERENCE,
            "context_length": context_length,
            "d_model": d_model,
            "swiglu_d": swiglu_d,
            "num_heads": num_heads,
            "num_key_value_heads": num_key_value_heads,
            "num_layers": num_layers,
            "rope_base": rope_base,
            "total_params": total_params,
        },

        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "training_dtype": str(training_dtype).replace("torch.", ""),
            "mixed_precision": use_mixed_precision,
            "eval_interval": eval_interval,
            "token_budget": TOKEN_BUDGET,
            "tokens_seen": tokens_seen,
            "final_step": step,
            "total_steps": total_steps_est,
            "checkpoint_dir": CHECKPOINT_DIR,
            "checkpoint_prefix": CHECKPOINT_PREFIX,
            "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
            "resumed_from": resume_checkpoint_path,
            "best_val_ppl": best_val_ppl,
            "final_train_ppl": train_ppl,
            "final_val_ppl": val_ppl,
        },

        "generation": generation_outputs,
    }

    config_path = os.path.join(run_dir, f"run_config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved → {config_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model",       type=int,   default=None)
    parser.add_argument("--num_layers",    type=int,   default=None)
    parser.add_argument("--num_heads",     type=int,   default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--swiglu_d",      type=int,   default=None)
    parser.add_argument("--rope_base",     type=float, default=None)
    parser.add_argument("--num_docs",      type=int,   default=None)
    parser.add_argument("--vocab_size",    type=int,   default=None)
    parser.add_argument("--token_budget",  type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--tokenizer_workers", type=int, default=None)
    parser.add_argument("--encode_workers",    type=int, default=None)
    args = parser.parse_args()

    # Override globals only if provided
    if args.d_model       is not None: d_model       = args.d_model
    if args.num_layers    is not None: num_layers    = args.num_layers
    if args.num_heads     is not None: num_heads     = args.num_heads
    if args.num_key_value_heads is not None: num_key_value_heads = args.num_key_value_heads
    if args.swiglu_d      is not None: swiglu_d      = args.swiglu_d
    if args.rope_base     is not None: rope_base     = args.rope_base
    if args.num_docs      is not None: NUM_DOCS      = args.num_docs
    if args.vocab_size    is not None: VOCAB_SIZE    = args.vocab_size
    if args.token_budget  is not None: TOKEN_BUDGET  = args.token_budget
    if args.learning_rate is not None: learning_rate = args.learning_rate
    if args.tokenizer_workers is not None: TOKENIZER_WORKERS = args.tokenizer_workers
    if args.encode_workers    is not None: ENCODE_WORKERS    = args.encode_workers

    main()
