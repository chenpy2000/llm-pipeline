import argparse
import csv
import json
import os
import re
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from dataset import HFCausalLMDataset
from mix_dataset import build_or_load_mixed_tokenized_blocks
from tokenizer import Tokenizer
from transformer import Decoder


# System
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_count = os.cpu_count() or 1
num_workers = 4
ENCODE_WORKERS = max(1, cpu_count - 1)

# Data
TOKENIZED_DATA_DIR = "./data/tokenized_cpt"
RAW_PARQUET_DIR = "./data/raw_parquet_cpt"
TOKENIZED_DATA_LABEL = None
TOKENIZE_BATCH_SIZE = 500
VOCAB_SIZE = 32768
SPECIAL_TOKENS = ["<|endoftext|>"]
VAL_TOKENS = 262_144
CHECKPOINT_INTERVAL_TOKENS = 1_000_000_000
CHECKPOINT_DIR = "./checkpoints/pretrain"
CHECKPOINT_PREFIX = "qwen25_coder_05b"

# Set this to a checkpoint filename or path to start CPT from a specific model.
# Examples:
# CHECKPOINT_NAME = "qwen25_coder_05b_7b.pt"
# CHECKPOINT_NAME = "output/20260706_123456/model_20260706_123456.pt"
CHECKPOINT_NAME = None

HF_TOKEN = None
RESUME_TRAINING_STATE = False

CPT_SOURCES = [
    {
        "label": "stack_edu_python",
        "dataset_id": "HuggingFaceTB/stack-edu",
        "config_name": "Python",
        "split": "train",
        "text_column": "text",
        "blob_id_column": "blob_id",
        "token_budget": 2_750_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "codeparrot_clean",
        "dataset_id": "codeparrot/codeparrot-clean",
        "config_name": "default",
        "split": "train",
        "text_column": "content",
        "file_format": "json",
        "token_budget": 750_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "starcoder_python",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "python",
        "split": "train",
        "text_column": "content",
        "token_budget": 250_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "starcoder_jupyter_scripts",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "jupyter-scripts-dedup-filtered",
        "split": "train",
        "text_column": "content",
        "token_budget": 100_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "starcoder_jupyter_structured",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "jupyter-structured-clean-dedup",
        "split": "train",
        "text_columns": ["content", "text", "code", "markdown"],
        "token_budget": 50_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "starcoder_github_issues",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "github-issues-filtered-structured",
        "split": "train",
        "text_columns": ["title", "body", "comments", "text", "content"],
        "token_budget": 50_000_000,
        "fim_rate": 0.0,
    },
    {
        "label": "starcoder_git_commits",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "git-commits-cleaned",
        "split": "train",
        "text_columns": ["message", "content", "diff", "text"],
        "token_budget": 50_000_000,
        "fim_rate": 0.15,
    },
    {
        "label": "opc_fineweb_code",
        "dataset_id": "OpenCoder-LLM/opc-fineweb-code-corpus",
        "config_name": "default",
        "split": "train",
        "text_column": "text",
        "token_budget": 400_000_000,
        "fim_rate": 0.30,
    },
    {
        "label": "finemath_4plus",
        "dataset_id": "HuggingFaceTB/finemath",
        "config_name": "finemath-4plus",
        "split": "train",
        "text_column": "text",
        "token_budget": 300_000_000,
        "fim_rate": 0.0,
    },
    {
        "label": "fineweb_edu_replay",
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "config_name": "sample-10BT",
        "split": "train",
        "text_column": "text",
        "token_budget": 300_000_000,
        "fim_rate": 0.0,
    },
]
TOKEN_BUDGET = sum(source["token_budget"] for source in CPT_SOURCES)

# Model
ARCHITECTURE_REFERENCE = "Qwen2.5-Coder-0.5B"
context_length = 1024
d_model = 896
swiglu_d = 4864
num_heads = 14
num_key_value_heads = 2
num_layers = 24
rope_base = 1_000_000.0

# Training
batch_size = 16
learning_rate = 3e-4
eval_interval = 1000
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
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        with bf16_autocast():
            loss = decoderLMmodel(X, Y)
        losses.append(loss.item())

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()

    decoderLMmodel.train()
    return perplexity


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.1):
    model.eval()
    token_ids = tokenizer.encode_ids(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_length:]
        with bf16_autocast():
            logits = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    model.train()
    return tokenizer.decode(x.squeeze(0).tolist())


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


def resolve_checkpoint_path(checkpoint_name):
    if checkpoint_name:
        candidates = [checkpoint_name]
        candidates.append(os.path.join(CHECKPOINT_DIR, checkpoint_name))
        if not checkpoint_name.endswith(".pt"):
            candidates.append(f"{checkpoint_name}.pt")
            candidates.append(os.path.join(CHECKPOINT_DIR, f"{checkpoint_name}.pt"))

        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        target_names = {os.path.basename(candidate) for candidate in candidates}
        matches = []
        for search_root in (CHECKPOINT_DIR, "output"):
            if not os.path.exists(search_root):
                continue
            for root, _, filenames in os.walk(search_root):
                for target_name in target_names:
                    if target_name in filenames:
                        matches.append(os.path.abspath(os.path.join(root, target_name)))
        matches = sorted(set(matches))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(
                f"Checkpoint name {checkpoint_name!r} is ambiguous: {matches}"
            )

        raise FileNotFoundError(
            f"Could not find checkpoint {checkpoint_name!r} as a path, under "
            f"{CHECKPOINT_DIR}, or under output/"
        )

    latest = find_latest_checkpoint()
    if latest is None:
        raise FileNotFoundError(
            "CPT needs an existing model checkpoint. Pass --checkpoint_name or create a pretrain checkpoint first."
        )
    _, path = latest
    return os.path.abspath(path)


def load_sources_json(path):
    with open(path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("CPT sources JSON must be a list of source objects")
    return sources


def model_config(total_params):
    return {
        "architecture_reference": ARCHITECTURE_REFERENCE,
        "context_length": context_length,
        "d_model": d_model,
        "swiglu_d": swiglu_d,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "num_layers": num_layers,
        "rope_base": rope_base,
        "vocab_size": VOCAB_SIZE,
        "total_params": total_params,
    }


def training_config(total_steps_est):
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_dtype": str(training_dtype).replace("torch.", ""),
        "mixed_precision": use_mixed_precision,
        "eval_interval": eval_interval,
        "token_budget": TOKEN_BUDGET,
        "total_steps": total_steps_est,
        "checkpoint_interval_tokens": CHECKPOINT_INTERVAL_TOKENS,
    }


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    checkpoint_dir,
    timestamp,
    checkpoint_index,
    step,
    tokens_seen,
    best_val_ppl,
    no_improve,
    total_steps_est,
    total_params,
    source_checkpoint_path,
    data_manifest,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{timestamp}_{checkpoint_index}b.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "cpt_tokens_seen": tokens_seen,
            "checkpoint_index": checkpoint_index,
            "best_val_ppl": best_val_ppl,
            "no_improve": no_improve,
            "total_steps": total_steps_est,
            "total_params": total_params,
            "source_checkpoint": source_checkpoint_path,
            "data_manifest": data_manifest,
            "model_config": model_config(total_params),
            "training_config": training_config(total_steps_est),
        },
        path,
    )
    print(f"CPT checkpoint saved -> {path} ({tokens_seen:,} tokens)")
    return path


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", f"cpt_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output -> {run_dir}")

    print("Loading Tokenizer")
    tokenizer_path = f"tokenizer/tokenizer_{VOCAB_SIZE}.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"CPT expects an existing tokenizer at {tokenizer_path}. Run pretraining/tokenizer setup first."
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")

    num_proc = max(1, ENCODE_WORKERS)
    token_blocks, data_manifest = build_or_load_mixed_tokenized_blocks(
        sources=CPT_SOURCES,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        context_length=context_length,
        tokenized_root=TOKENIZED_DATA_DIR,
        raw_data_dir=RAW_PARQUET_DIR,
        data_label=TOKENIZED_DATA_LABEL,
        hf_token=HF_TOKEN,
        encode_workers=num_proc,
        tokenize_batch_size=TOKENIZE_BATCH_SIZE,
    )

    block_dataset = HFCausalLMDataset(token_blocks, context_length)
    if len(block_dataset) < 2:
        raise ValueError("Need at least two token blocks to create train and val splits")
    total_tokens = data_manifest["total_training_tokens"]
    token_ids = range(total_tokens)

    total_tokens = len(token_ids)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Tokenized dataset label: {data_manifest['label']}")
    for source_info in data_manifest["sources"]:
        print(
            f"  {source_info['label']}: "
            f"{source_info['blocks']:,} blocks, "
            f"{source_info['training_tokens']:,} training tokens"
        )

    val_samples = max(1, VAL_TOKENS // context_length)
    val_samples = min(val_samples, max(1, len(block_dataset) // 10))
    val_samples = min(val_samples, len(block_dataset) - 1)
    split = len(block_dataset) - val_samples
    train_dataset = Subset(block_dataset, range(split))
    val_dataset = Subset(block_dataset, range(split, len(block_dataset)))

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

    model = Decoder(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_head=num_heads,
        n_kv_head=num_key_value_heads,
        swiglu_d=swiglu_d,
        n_layer=num_layers,
        rope_base=rope_base,
    )

    print("Model Summary:")
    print(
        f"  Layers: {num_layers} | Q Heads: {num_heads} | "
        f"KV Heads: {num_key_value_heads} | Context: {context_length}"
    )
    print(f"  d_model: {d_model} | swiglu_d: {swiglu_d} | RoPE base: {rope_base:g}")
    print(
        f"  Training dtype: {str(training_dtype).replace('torch.', '')} | "
        f"Mixed precision: {use_mixed_precision}"
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    tokens_per_step = batch_size * context_length
    total_steps_est = (
        (TOKEN_BUDGET + tokens_per_step - 1) // tokens_per_step
        if TOKEN_BUDGET > 0
        else ((len(train_dataset) + batch_size - 1) // batch_size)
    )
    total_steps_est = max(1, total_steps_est)

    warmup_steps = max(1, int(0.02 * total_steps_est))
    decay_steps = max(1, total_steps_est - warmup_steps)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-5,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=learning_rate * 0.1,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    step = 0
    best_val_ppl = float("inf")
    no_improve = 0
    tokens_seen = 0
    source_checkpoint_path = resolve_checkpoint_path(CHECKPOINT_NAME)
    save_dir = os.path.dirname(source_checkpoint_path)

    print(f"Loading model weights from {source_checkpoint_path} ...")
    checkpoint = torch.load(source_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if RESUME_TRAINING_STATE:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("cpt_tokens_seen", checkpoint.get("tokens_seen", 0)))
        best_val_ppl = checkpoint.get("best_val_ppl", best_val_ppl)
        no_improve = int(checkpoint.get("no_improve", 0))
        print(f"Resuming CPT state at step {step:,}, tokens_seen={tokens_seen:,}")
    else:
        print("Loaded model weights only; CPT optimizer and LR schedule start fresh")

    start_sample = min(tokens_seen // context_length, len(train_dataset))
    if start_sample > 0:
        print(f"Skipping {start_sample:,} already-seen CPT training samples")
    train_subset = Subset(train_dataset, range(start_sample, len(train_dataset)))
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    next_checkpoint_index = tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1

    log_path = os.path.join(run_dir, f"training_log_{timestamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_steps", "loss", "train_ppl", "val_ppl", "lr"])

    print("CPT training ...")
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
            desc="CPT",
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
                    val_ppl = compute_perplexity(model, val_loader)
                    progress_bar.write(
                        f"Step {step}/{total_steps_est} | "
                        f"Tokens: {tokens_seen:,} | "
                        f"LR: {current_lr:.2e} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Train PPL: {train_ppl:.2f} | "
                        f"Val PPL: {val_ppl:.2f}"
                    )

                    log_writer.writerow(
                        [
                            step,
                            total_steps_est,
                            f"{loss.item():.6f}",
                            f"{train_ppl:.4f}",
                            f"{val_ppl:.4f}",
                            f"{current_lr:.6e}",
                        ]
                    )
                    log_file.flush()

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
                        checkpoint_dir=save_dir,
                        timestamp=timestamp,
                        checkpoint_index=next_checkpoint_index,
                        step=step,
                        tokens_seen=tokens_seen,
                        best_val_ppl=best_val_ppl,
                        no_improve=no_improve,
                        total_steps_est=total_steps_est,
                        total_params=total_params,
                        source_checkpoint_path=source_checkpoint_path,
                        data_manifest=data_manifest,
                    )
                    next_checkpoint_index += 1

                if TOKEN_BUDGET > 0 and tokens_seen >= TOKEN_BUDGET:
                    progress_bar.write(f"Token budget reached at step {step} ({tokens_seen:,} tokens)")
                    break
            else:
                if TOKEN_BUDGET > 0 and tokens_seen < TOKEN_BUDGET:
                    progress_bar.write(
                        f"CPT data exhausted at {tokens_seen:,} tokens before token budget {TOKEN_BUDGET:,}"
                    )

    log_file.close()

    val_ppl = compute_perplexity(model, val_loader)
    if train_ppl is None:
        print(f"Final - Val PPL: {val_ppl:.2f}")
    else:
        print(f"Final - Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    model_path = os.path.join(save_dir, f"{timestamp}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "tokens_seen": tokens_seen,
            "cpt_tokens_seen": tokens_seen,
            "best_val_ppl": best_val_ppl,
            "no_improve": no_improve,
            "source_checkpoint": source_checkpoint_path,
            "data_manifest": data_manifest,
            "model_config": model_config(total_params),
            "training_config": training_config(total_steps_est),
        },
        model_path,
    )
    print(f"CPT model saved -> {model_path}")

    prompts = [
        "def fibonacci(n):",
        "import torch\n\nclass",
        "The bug happens because",
    ]
    print("\n--- Generation ---")
    generation_outputs = []
    for prompt in prompts:
        output = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")
        generation_outputs.append({"prompt": prompt, "output": output})

    run_config = {
        "timestamp": timestamp,
        "device": str(device),
        "data": {
            "tokenized_data_dir": TOKENIZED_DATA_DIR,
            "tokenized_data_label": data_manifest["label"],
            "tokenized_manifest_path": os.path.join(
                TOKENIZED_DATA_DIR,
                data_manifest["label"],
                "manifest.json",
            ),
            "raw_parquet_dir": RAW_PARQUET_DIR,
            "total_tokens": total_tokens,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_tokens": val_samples * context_length,
            "sources": data_manifest["sources"],
            "encode_workers": num_proc,
            "tokenize_batch_size": TOKENIZE_BATCH_SIZE,
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "tokenizer_path": tokenizer_path,
        },
        "model": model_config(total_params),
        "training": {
            **training_config(total_steps_est),
            "tokens_seen": tokens_seen,
            "final_step": step,
            "checkpoint_dir": save_dir,
            "source_checkpoint": source_checkpoint_path,
            "final_checkpoint": model_path,
            "best_val_ppl": best_val_ppl,
            "final_train_ppl": train_ppl,
            "final_val_ppl": val_ppl,
            "resume_training_state": RESUME_TRAINING_STATE,
        },
        "generation": generation_outputs,
    }

    config_path = os.path.join(run_dir, f"run_config_{timestamp}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved -> {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--sources_json", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--swiglu_d", type=int, default=None)
    parser.add_argument("--rope_base", type=float, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--encode_workers", type=int, default=None)
    parser.add_argument("--tokenized_data_dir", type=str, default=None)
    parser.add_argument("--raw_parquet_dir", type=str, default=None)
    parser.add_argument("--data_label", type=str, default=None)
    parser.add_argument("--tokenize_batch_size", type=int, default=None)
    parser.add_argument("--val_tokens", type=int, default=None)
    parser.add_argument("--resume_training_state", action="store_true")
    args = parser.parse_args()

    if args.sources_json is not None:
        CPT_SOURCES = load_sources_json(args.sources_json)
        if args.token_budget is None:
            TOKEN_BUDGET = sum(int(source["token_budget"]) for source in CPT_SOURCES)

    if args.checkpoint_name is not None:
        CHECKPOINT_NAME = args.checkpoint_name
    if args.hf_token is not None:
        HF_TOKEN = args.hf_token
    if args.d_model is not None:
        d_model = args.d_model
    if args.num_layers is not None:
        num_layers = args.num_layers
    if args.num_heads is not None:
        num_heads = args.num_heads
    if args.num_key_value_heads is not None:
        num_key_value_heads = args.num_key_value_heads
    if args.swiglu_d is not None:
        swiglu_d = args.swiglu_d
    if args.rope_base is not None:
        rope_base = args.rope_base
    if args.vocab_size is not None:
        VOCAB_SIZE = args.vocab_size
    if args.token_budget is not None:
        TOKEN_BUDGET = args.token_budget
    if args.learning_rate is not None:
        learning_rate = args.learning_rate
    if args.encode_workers is not None:
        ENCODE_WORKERS = args.encode_workers
    if args.tokenized_data_dir is not None:
        TOKENIZED_DATA_DIR = args.tokenized_data_dir
    if args.raw_parquet_dir is not None:
        RAW_PARQUET_DIR = args.raw_parquet_dir
    if args.data_label is not None:
        TOKENIZED_DATA_LABEL = args.data_label
    if args.tokenize_batch_size is not None:
        TOKENIZE_BATCH_SIZE = args.tokenize_batch_size
    if args.val_tokens is not None:
        VAL_TOKENS = args.val_tokens
    if args.resume_training_state:
        RESUME_TRAINING_STATE = True

    main()
