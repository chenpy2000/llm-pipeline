import torch
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import os
import re
import csv
import json
import argparse
from datetime import datetime

from tqdm.auto import tqdm

from config import get_config_section, torch_dtype
from main_tokenizer import (
    default_tokenizer_path,
    load_required_tokenizer,
    validate_checkpoint_vocab,
)
from dataset import build_causal_training_data

from transformer import Decoder

TOKENIZER_CONFIG = get_config_section("tokenizer")
MODEL_CONFIG = get_config_section("model")
TRAINING_CONFIG = get_config_section("training")

# System
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_count   = os.cpu_count() or 1
num_workers = 4
ENCODE_WORKERS    = max(1, cpu_count - 1)

# Data
DATASET_ID     = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"
DATASET_SPLIT  = "train"
TEXT_COLUMN    = "text"
TOKENIZED_DATA_DIR = "./data/tokenized"
RAW_DATA_DIR        = "./data/raw_parquet"
TOKENIZED_DATA_LABEL = None
TOKENIZE_BATCH_SIZE  = 500
NUM_DOCS       = 9_672_101  # Max documents in FineWeb-EDU sample-10BT.
VOCAB_SIZE     = int(TOKENIZER_CONFIG["vocab_size"])
SPECIAL_TOKENS = list(TOKENIZER_CONFIG["special_tokens"])
TOKEN_BUDGET   = 7_000_000_000 # 0 = disabled (epoch mode), >0 = token-budget mode
PRETRAIN_SOURCES = None
VAL_TOKENS     = int(TRAINING_CONFIG["val_tokens"])    # validation token budget
CHECKPOINT_INTERVAL_TOKENS = 1_000_000_000
CHECKPOINT_DIR    = "./checkpoints/pretrain"
CHECKPOINT_PREFIX = "qwen25_coder_05b_v151936"

# Model
ARCHITECTURE_REFERENCE = MODEL_CONFIG["architecture_reference"]
context_length = int(MODEL_CONFIG["context_length"])      # maximum sequence length
d_model        = int(MODEL_CONFIG["d_model"])        # embedding dimension
swiglu_d       = int(MODEL_CONFIG["swiglu_d"])       # SwiGLU hidden dimension
num_heads      = int(MODEL_CONFIG["num_heads"])         # number of attention heads
num_key_value_heads = int(MODEL_CONFIG["num_key_value_heads"])     # number of grouped-query attention KV heads
num_layers     = int(MODEL_CONFIG["num_layers"])     # number of transformer layers
rope_base      = float(MODEL_CONFIG["rope_base"])

# Training
batch_size     = int(TRAINING_CONFIG["batch_size"])
learning_rate  = float(TRAINING_CONFIG["learning_rate"])
eval_interval  = int(TRAINING_CONFIG["eval_interval"])    # log every N steps
training_dtype = torch_dtype(TRAINING_CONFIG["dtype"])
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
    token_ids = tokenizer.encode_ids(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to block_size if the sequence gets too long
        x_cond = x[:, -context_length:]
        with bf16_autocast():
            logits = model(x_cond)                    # no targets -> returns logits
        logits = logits[:, -1, :] / temperature       # last position only
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    model.train()
    return tokenizer.decode(x.squeeze(0).tolist())

def checkpoint_path(checkpoint_index):
    return os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_{checkpoint_index}b.pt")


def load_sources_json(path):
    with open(path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("Pretrain sources JSON must be a list of source objects")
    return sources


def pretrain_sources():
    if PRETRAIN_SOURCES is not None:
        return PRETRAIN_SOURCES

    return [
        {
            "label": "fineweb_edu",
            "dataset_id": DATASET_ID,
            "config_name": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "text_column": TEXT_COLUMN,
            "token_budget": TOKEN_BUDGET,
            "num_docs": NUM_DOCS,
            "file_format": "parquet",
        }
    ]

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

def save_training_checkpoint(model, optimizer, scheduler, checkpoint_index, step,
                             tokens_seen, best_val_ppl, no_improve, total_steps_est, total_params):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = checkpoint_path(checkpoint_index)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
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

    # Timestamp and output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join("output", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output -> {run_dir}")

    print("Loading Tokenizer")
    tokenizer_path = default_tokenizer_path(VOCAB_SIZE)
    tokenizer = load_required_tokenizer(
        tokenizer_path=tokenizer_path,
        expected_vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")

    # Encode cached token blocks

    num_proc = max(1, ENCODE_WORKERS)
    training_data = build_causal_training_data(
        sources=pretrain_sources(),
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        context_length=context_length,
        tokenized_root=TOKENIZED_DATA_DIR,
        raw_data_dir=RAW_DATA_DIR,
        data_label=TOKENIZED_DATA_LABEL,
        val_tokens=VAL_TOKENS,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        encode_workers=num_proc,
        tokenize_batch_size=TOKENIZE_BATCH_SIZE,
    )

    data_manifest = training_data.data_manifest
    train_dataset = training_data.train_dataset
    val_dataset = training_data.val_dataset
    val_loader = training_data.val_loader
    total_tokens = training_data.total_tokens
    print(f"Total tokens: {total_tokens:,}")

    print(f"Tokenized dataset label: {data_manifest['label']}")

    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

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

    # Cosine LR scheduler (matched to token budget, or one epoch if no budget)
    tokens_per_step = batch_size * context_length
    total_steps_est = ((TOKEN_BUDGET + tokens_per_step - 1) // tokens_per_step) if TOKEN_BUDGET > 0 else ((len(train_dataset) + batch_size - 1) // batch_size)
    
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

    if resume_checkpoint is None:
        print("No pretrain checkpoint found; training from scratch")
    else:
        checkpoint_index, resume_checkpoint_path = resume_checkpoint
        print(f"Loading checkpoint {checkpoint_index}b from {resume_checkpoint_path} ...")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        validate_checkpoint_vocab(
            checkpoint=checkpoint,
            expected_vocab_size=tokenizer.vocab_size,
            checkpoint_path=resume_checkpoint_path,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", step * tokens_per_step))
        best_val_ppl = checkpoint.get("best_val_ppl", best_val_ppl)
        no_improve = int(checkpoint.get("no_improve", 0))
        print(f"Resuming at step {step:,}, tokens_seen={tokens_seen:,}")

    start_sample = min(tokens_seen // context_length, len(train_dataset))
    if start_sample > 0:
        print(f"Skipping {start_sample:,} already-seen training samples")
    train_loader = training_data.make_train_loader(start_sample=start_sample)
    next_checkpoint_index = tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1

    # Training log CSV
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
        progress_total_tokens = (
            TOKEN_BUDGET
            if TOKEN_BUDGET > 0
            else len(train_dataset) * context_length
        )
        progress_initial_tokens = min(tokens_seen, progress_total_tokens)
        with tqdm(
            total=progress_total_tokens,
            initial=progress_initial_tokens,
            desc="Training",
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress_bar:
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
                current_lr = optimizer.param_groups[0]["lr"]

                progress_bar.update(
                    max(0, min(tokens_seen, progress_total_tokens) - progress_bar.n)
                )
                progress_bar.set_postfix(
                    step=f"{step:,}/{total_steps_est:,}",
                    loss=f"{loss.item():.4f}",
                    ppl=f"{train_ppl:.2f}",
                    lr=f"{current_lr:.2e}",
                    refresh=False,
                )

                if step % eval_interval == 0:
                    val_ppl   = compute_perplexity(model, val_loader)
                    progress_bar.write(
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
                    progress_bar.write(f"Token budget reached at step {step} ({tokens_seen:,} tokens)")
                    break
            else:
                if TOKEN_BUDGET > 0 and tokens_seen < TOKEN_BUDGET:
                    progress_bar.write(
                        f"Training data exhausted at {tokens_seen:,} tokens before token budget {TOKEN_BUDGET:,}"
                    )

    log_file.close()

    # Final eval
    val_ppl   = compute_perplexity(model, val_loader)
    if train_ppl is None:
        print(f"Final - Val PPL: {val_ppl:.2f}")
    else:
        print(f"Final - Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # Save model
    model_path = os.path.join(run_dir, f"model_{timestamp}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_ppl": best_val_ppl,
        "no_improve": no_improve,
    }, model_path)
    print(f"Model saved -> {model_path}")

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

    # Save run config
    run_config = {
        "timestamp": timestamp,
        "device": str(device),

        "data": {
            "total_tokens": total_tokens,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_tokens": training_data.val_samples * context_length,
            "encode_workers": num_proc,
            "tokenize_batch_size": TOKENIZE_BATCH_SIZE,
            "tokenized_data_dir": TOKENIZED_DATA_DIR,
            "tokenized_data_label": data_manifest["label"],
            "tokenized_manifest_path": os.path.join(
                TOKENIZED_DATA_DIR,
                data_manifest["label"],
                "manifest.json",
            ),
            "raw_data_dir": RAW_DATA_DIR,
            "sources": data_manifest["sources"],
        },

        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
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
    print(f"Config saved -> {config_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model",       type=int,   default=None)
    parser.add_argument("--num_layers",    type=int,   default=None)
    parser.add_argument("--num_heads",     type=int,   default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--swiglu_d",      type=int,   default=None)
    parser.add_argument("--rope_base",     type=float, default=None)
    parser.add_argument("--num_docs",      type=int,   default=None)
    parser.add_argument("--sources_json",  type=str,   default=None)
    parser.add_argument("--vocab_size",    type=int,   default=None)
    parser.add_argument("--token_budget",  type=int,   default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--encode_workers",    type=int, default=None)
    parser.add_argument("--tokenized_data_dir", type=str, default=None)
    parser.add_argument("--raw_data_dir", "--raw_parquet_dir", dest="raw_data_dir", type=str, default=None)
    parser.add_argument("--data_label",         type=str, default=None)
    parser.add_argument("--tokenize_batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.sources_json is not None:
        PRETRAIN_SOURCES = load_sources_json(args.sources_json)
        if args.token_budget is None:
            TOKEN_BUDGET = sum(int(source["token_budget"]) for source in PRETRAIN_SOURCES)

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
    if args.encode_workers    is not None: ENCODE_WORKERS    = args.encode_workers
    if args.tokenized_data_dir is not None: TOKENIZED_DATA_DIR = args.tokenized_data_dir
    if args.raw_data_dir       is not None: RAW_DATA_DIR       = args.raw_data_dir
    if args.data_label         is not None: TOKENIZED_DATA_LABEL = args.data_label
    if args.tokenize_batch_size is not None: TOKENIZE_BATCH_SIZE = args.tokenize_batch_size

    main()
