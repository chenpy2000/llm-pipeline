from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

from config import get_config_section, torch_dtype
from main_tokenizer import (
    default_tokenizer_path,
    load_required_tokenizer,
    validate_checkpoint_vocab,
)
from dataset import (
    CHAT_END,
    CHAT_START,
    HFSFTMaskedDataset,
    IGNORE_INDEX,
    LABEL_BLOCK_COLUMN,
    build_training_data,
    build_or_load_mixed_sft_tokenized_blocks,
    set_sft_stage_name,
)
from transformer import Decoder

TOKENIZER_CONFIG = get_config_section("tokenizer")
MODEL_CONFIG = get_config_section("model")
TRAINING_CONFIG = get_config_section("training")

# System
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_count = os.cpu_count() or 1
num_workers = 4
ENCODE_WORKERS = max(1, cpu_count - 1)

# Data
STAGE_NAME = "sft_qa_cot"
TOKENIZED_DATA_DIR = "./data/tokenized_sft_qa_cot"
TOKENIZED_DATA_LABEL = None
TOKENIZED_SHARD_BLOCKS = 8192
VOCAB_SIZE = int(TOKENIZER_CONFIG["vocab_size"])
SPECIAL_TOKENS = list(TOKENIZER_CONFIG["special_tokens"])
VAL_TOKENS = int(TRAINING_CONFIG["val_tokens"])
CHECKPOINT_INTERVAL_TOKENS = 1_000_000_000
CHECKPOINT_DIR = "./checkpoints/pretrain"
CHECKPOINT_PREFIX = "qwen25_coder_05b_v151936"

# Set this to a checkpoint filename or path to start SFT from a specific model.
# For the coding SFT run, pass the final QA/CoT SFT checkpoint with --checkpoint_name.
CHECKPOINT_NAME = None

HF_TOKEN = None
RESUME_TRAINING_STATE = False



SFT_SOURCES = [
    {
        "label": "smol_smoltalk",
        "dataset_id": "HuggingFaceTB/smol-smoltalk",
        "split": "train",
        "messages_column": "messages",
        "token_budget": 18_000_000,
    },
    {
        "label": "ultrachat_200k",
        "dataset_id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "messages_column": "messages",
        "token_budget": 8_000_000,
    },
    {
        "label": "open_orca",
        "dataset_id": "Open-Orca/OpenOrca",
        "split": "train",
        "system_column": "system_prompt",
        "user_columns": ["question"],
        "assistant_column": "response",
        "token_budget": 10_000_000,
    },
    {
        "label": "openr1_math_220k",
        "dataset_id": "open-r1/OpenR1-Math-220k",
        "config_name": "default",
        "split": "train",
        "messages_column": "messages",
        "token_budget": 17_000_000,
    },
    {
        "label": "mixture_of_thoughts_all",
        "dataset_id": "open-r1/Mixture-of-Thoughts",
        "config_name": "all",
        "split": "train",
        "messages_column": "messages",
        "token_budget": 25_000_000,
    },
    {
        "label": "smol_smoltalk_final",
        "dataset_id": "HuggingFaceTB/smol-smoltalk",
        "split": "train",
        "skip_examples": 400_000,
        "messages_column": "messages",
        "token_budget": 2_000_000,
    },
]
TOKEN_BUDGET = sum(source["token_budget"] for source in SFT_SOURCES)

# Model
ARCHITECTURE_REFERENCE = MODEL_CONFIG["architecture_reference"]
context_length = int(MODEL_CONFIG["context_length"])
d_model = int(MODEL_CONFIG["d_model"])
swiglu_d = int(MODEL_CONFIG["swiglu_d"])
num_heads = int(MODEL_CONFIG["num_heads"])
num_key_value_heads = int(MODEL_CONFIG["num_key_value_heads"])
num_layers = int(MODEL_CONFIG["num_layers"])
rope_base = float(MODEL_CONFIG["rope_base"])

# Training
batch_size = int(TRAINING_CONFIG["batch_size"])
learning_rate = float(TRAINING_CONFIG["learning_rate"])
eval_interval = int(TRAINING_CONFIG["eval_interval"])
training_dtype = torch_dtype(TRAINING_CONFIG["dtype"])
use_mixed_precision = device.type == "cuda" and torch.cuda.is_bf16_supported()

GENERATION_PROMPTS = [
    f"{CHAT_START}user\nExplain why careful source checking matters in scientific reasoning.{CHAT_END}\n{CHAT_START}assistant\n",
    f"{CHAT_START}user\nI feel stuck on a hard math problem. Can you help me approach it calmly?{CHAT_END}\n{CHAT_START}assistant\n",
    f"{CHAT_START}user\nA train leaves at noon traveling 60 mph. Another leaves at 1pm traveling 90 mph. When does it catch up?{CHAT_END}\n{CHAT_START}assistant\n",
]


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
    token_ids = tokenizer.encode_ids(prompt, add_special_tokens=False)
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
            "SFT needs an existing model checkpoint. Pass --checkpoint_name or create a checkpoint first."
        )
    _, path = latest
    return os.path.abspath(path)


def load_sources_json(path):
    with open(path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("SFT sources JSON must be a list of source objects")
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
    path = os.path.join(checkpoint_dir, f"{timestamp}_{STAGE_NAME}_{checkpoint_index}b.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "sft_tokens_seen": tokens_seen,
        "checkpoint_index": checkpoint_index,
        "stage_name": STAGE_NAME,
        "best_val_ppl": best_val_ppl,
        "no_improve": no_improve,
        "total_steps": total_steps_est,
        "total_params": total_params,
        "source_checkpoint": source_checkpoint_path,
        "data_manifest": data_manifest,
        "model_config": model_config(total_params),
        "training_config": training_config(total_steps_est),
    }
    payload[f"{STAGE_NAME}_tokens_seen"] = tokens_seen
    torch.save(payload, path)
    print(f"SFT checkpoint saved -> {path} ({tokens_seen:,} tokens)")
    return path


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", f"{STAGE_NAME}_{timestamp}")
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

    set_sft_stage_name(STAGE_NAME)
    token_blocks, data_manifest = build_or_load_mixed_sft_tokenized_blocks(
        sources=SFT_SOURCES,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        context_length_value=context_length,
        tokenized_root=TOKENIZED_DATA_DIR,
        data_label=TOKENIZED_DATA_LABEL,
        hf_token=HF_TOKEN,
        tokenized_shard_blocks=TOKENIZED_SHARD_BLOCKS,
    )
    training_data = build_training_data(
        token_blocks=token_blocks,
        data_manifest=data_manifest,
        dataset_cls=HFSFTMaskedDataset,
        context_length=context_length,
        val_tokens=VAL_TOKENS,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    data_manifest = training_data.data_manifest
    train_dataset = training_data.train_dataset
    val_dataset = training_data.val_dataset
    val_loader = training_data.val_loader
    total_tokens = training_data.total_tokens

    print(f"Total tokens: {total_tokens:,}")
    print(f"Tokenized dataset label: {data_manifest['label']}")
    for source_info in data_manifest["sources"]:
        print(
            f"  {source_info['label']}: "
            f"{source_info['blocks']:,} blocks, "
            f"{source_info['training_tokens']:,} training tokens"
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
    validate_checkpoint_vocab(
        checkpoint=checkpoint,
        expected_vocab_size=tokenizer.vocab_size,
        checkpoint_path=source_checkpoint_path,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if RESUME_TRAINING_STATE:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get(f"{STAGE_NAME}_tokens_seen", checkpoint.get("sft_tokens_seen", 0)))
        best_val_ppl = checkpoint.get("best_val_ppl", best_val_ppl)
        no_improve = int(checkpoint.get("no_improve", 0))
        print(f"Resuming SFT state at step {step:,}, tokens_seen={tokens_seen:,}")
    else:
        print("Loaded model weights only; SFT optimizer and LR schedule start fresh")

    start_sample = min(tokens_seen // context_length, len(train_dataset))
    if start_sample > 0:
        print(f"Skipping {start_sample:,} already-seen SFT training samples")
    train_loader = training_data.make_train_loader(start_sample=start_sample)
    next_checkpoint_index = tokens_seen // CHECKPOINT_INTERVAL_TOKENS + 1

    log_path = os.path.join(run_dir, f"training_log_{timestamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_steps", "loss", "train_ppl", "val_ppl", "lr"])

    print(f"{STAGE_NAME} training ...")
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
            desc=STAGE_NAME,
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
                        f"SFT data exhausted at {tokens_seen:,} tokens before token budget {TOKEN_BUDGET:,}"
                    )

    log_file.close()

    val_ppl = compute_perplexity(model, val_loader)
    if train_ppl is None:
        print(f"Final - Val PPL: {val_ppl:.2f}")
    else:
        print(f"Final - Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    model_path = os.path.join(save_dir, f"{timestamp}_{STAGE_NAME}.pt")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "sft_tokens_seen": tokens_seen,
        "best_val_ppl": best_val_ppl,
        "no_improve": no_improve,
        "stage_name": STAGE_NAME,
        "source_checkpoint": source_checkpoint_path,
        "data_manifest": data_manifest,
        "model_config": model_config(total_params),
        "training_config": training_config(total_steps_est),
    }
    payload[f"{STAGE_NAME}_tokens_seen"] = tokens_seen
    torch.save(payload, model_path)
    print(f"SFT model saved -> {model_path}")

    print("\n--- Generation ---")
    generation_outputs = []
    for prompt in GENERATION_PROMPTS:
        output = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")
        generation_outputs.append({"prompt": prompt, "output": output})

    run_config = {
        "timestamp": timestamp,
        "stage_name": STAGE_NAME,
        "device": str(device),
        "data": {
            "tokenized_data_dir": TOKENIZED_DATA_DIR,
            "tokenized_data_label": data_manifest["label"],
            "tokenized_manifest_path": os.path.join(
                TOKENIZED_DATA_DIR,
                data_manifest["label"],
                "manifest.json",
            ),
            "total_tokens": total_tokens,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "val_tokens": training_data.val_samples * context_length,
            "sources": data_manifest["sources"],
            "tokenized_shard_blocks": TOKENIZED_SHARD_BLOCKS,
            "label_column": LABEL_BLOCK_COLUMN,
            "ignore_index": IGNORE_INDEX,
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "tokenizer_path": tokenizer_path,
            "chat_start": CHAT_START,
            "chat_end": CHAT_END,
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


def parse_and_run():
    global SFT_SOURCES, TOKEN_BUDGET, CHECKPOINT_NAME, HF_TOKEN, d_model
    global num_layers, num_heads, num_key_value_heads, swiglu_d, rope_base
    global VOCAB_SIZE, learning_rate, ENCODE_WORKERS, TOKENIZED_DATA_DIR
    global TOKENIZED_DATA_LABEL, TOKENIZED_SHARD_BLOCKS, VAL_TOKENS
    global RESUME_TRAINING_STATE

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
    parser.add_argument("--data_label", type=str, default=None)
    parser.add_argument("--tokenized_shard_blocks", type=int, default=None)
    parser.add_argument("--val_tokens", type=int, default=None)
    parser.add_argument("--resume_training_state", action="store_true")
    args = parser.parse_args()

    if args.sources_json is not None:
        SFT_SOURCES = load_sources_json(args.sources_json)
        if args.token_budget is None:
            TOKEN_BUDGET = sum(int(source["token_budget"]) for source in SFT_SOURCES)

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
    if args.data_label is not None:
        TOKENIZED_DATA_LABEL = args.data_label
    if args.tokenized_shard_blocks is not None:
        TOKENIZED_SHARD_BLOCKS = args.tokenized_shard_blocks
    if args.val_tokens is not None:
        VAL_TOKENS = args.val_tokens
    if args.resume_training_state:
        RESUME_TRAINING_STATE = True

    main()


if __name__ == "__main__":
    parse_and_run()
