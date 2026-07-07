from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from data_pipeline import (
    TOKEN_BLOCK_COLUMN,
    file_sha256,
    load_tokenized_blocks,
    manifest_is_complete,
    sanitize_label,
    write_json_atomic,
)
from tokenizer import Tokenizer
from transformer import Decoder


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
VOCAB_SIZE = 32768
SPECIAL_TOKENS = ["<|endoftext|>"]
VAL_TOKENS = 262_144
CHECKPOINT_INTERVAL_TOKENS = 1_000_000_000
CHECKPOINT_DIR = "./checkpoints/pretrain"
CHECKPOINT_PREFIX = "qwen25_coder_05b"

# Set this to a checkpoint filename or path to start SFT from a specific model.
# For the coding SFT run, pass the final QA/CoT SFT checkpoint with --checkpoint_name.
CHECKPOINT_NAME = "output/20260706_040426/model_20260706_040426.pt"
# CHECKPOINT_NAME = None

HF_TOKEN = None
RESUME_TRAINING_STATE = False

CHAT_START = "<|im_start|>"
CHAT_END = "<|im_end|>"
LABEL_BLOCK_COLUMN = "labels"
IGNORE_INDEX = -100
SFT_TOKENIZED_PIPELINE_VERSION = 2


class HFSFTMaskedDataset(Dataset):
    """Wrap fixed-width token and label blocks for assistant-only SFT."""

    def __init__(
        self,
        hf_dataset,
        block_size,
        input_column=TOKEN_BLOCK_COLUMN,
        label_column=LABEL_BLOCK_COLUMN,
    ):
        self.hf_dataset = hf_dataset
        self.block_size = block_size
        self.input_column = input_column
        self.label_column = label_column

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        token_ids = row[self.input_column]
        labels = row[self.label_column]
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            labels = labels.to(dtype=torch.long)

        expected_width = self.block_size + 1
        if token_ids.numel() != expected_width:
            raise ValueError(
                f"Expected token block width {expected_width}, got {token_ids.numel()}"
            )
        if labels.numel() != expected_width:
            raise ValueError(
                f"Expected label block width {expected_width}, got {labels.numel()}"
            )

        return token_ids[:-1], labels[1:]


@dataclass(frozen=True)
class SFTSource:
    label: str
    dataset_id: str
    token_budget: int
    split: str = "train"
    config_name: str | None = None
    skip_examples: int = 0
    messages_column: str | None = None
    user_columns: tuple[str, ...] = ()
    assistant_column: str | None = None
    system_column: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)


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
ARCHITECTURE_REFERENCE = "Qwen2.5-Coder-0.5B"
context_length = 32768
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


def normalize_sft_source(source: SFTSource | dict[str, Any]) -> SFTSource:
    if isinstance(source, SFTSource):
        return source

    source = dict(source)
    user_columns = source.get("user_columns") or ()
    if isinstance(user_columns, str):
        user_columns = (user_columns,)
    else:
        user_columns = tuple(user_columns)

    return SFTSource(
        label=source["label"],
        dataset_id=source["dataset_id"],
        token_budget=int(source["token_budget"]),
        split=source.get("split", "train"),
        config_name=source.get("config_name"),
        skip_examples=int(source.get("skip_examples", 0)),
        messages_column=source.get("messages_column"),
        user_columns=user_columns,
        assistant_column=source.get("assistant_column"),
        system_column=source.get("system_column"),
        load_kwargs=dict(source.get("load_kwargs") or {}),
    )


def source_to_manifest(source: SFTSource) -> dict[str, Any]:
    return {
        "label": source.label,
        "dataset_id": source.dataset_id,
        "config_name": source.config_name,
        "split": source.split,
        "skip_examples": source.skip_examples,
        "messages_column": source.messages_column,
        "user_columns": list(source.user_columns),
        "assistant_column": source.assistant_column,
        "system_column": source.system_column,
        "token_budget": source.token_budget,
        "load_kwargs": dict(source.load_kwargs),
    }


def default_sft_mix_label(
    sources: list[SFTSource],
    vocab_size: int,
    context_length_value: int,
) -> str:
    digest_input = json.dumps(
        [source_to_manifest(source) for source in sources],
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:12]
    return sanitize_label(
        f"{STAGE_NAME}_mask_v{SFT_TOKENIZED_PIPELINE_VERSION}_mix_"
        f"{digest}_ctx{context_length_value}_v{vocab_size}"
    )


def sft_source_cache_label(
    source: SFTSource,
    vocab_size: int,
    context_length_value: int,
) -> str:
    digest = hashlib.sha1(
        json.dumps(source_to_manifest(source), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return sanitize_label(
        f"{source.label}_mask_v{SFT_TOKENIZED_PIPELINE_VERSION}_"
        f"{digest}_ctx{context_length_value}_v{vocab_size}"
    )


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, (list, tuple)):
        parts = [normalize_text_value(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_role(role: Any) -> str:
    role = normalize_text_value(role).strip().lower() or "user"
    if role not in {"system", "user", "assistant"}:
        role = "user"
    return role


def append_token_span(
    input_ids: list[int],
    labels: list[int],
    tokenizer,
    text: str,
    train_on_span: bool,
) -> None:
    ids = tokenizer.encode_ids(text, add_special_tokens=False)
    input_ids.extend(ids)
    if train_on_span:
        labels.extend(ids)
    else:
        labels.extend([IGNORE_INDEX] * len(ids))


def append_chat_turn(
    input_ids: list[int],
    labels: list[int],
    tokenizer,
    role: Any,
    content: Any,
) -> str | None:
    role = normalize_role(role)
    content = normalize_text_value(content).strip()
    if not content:
        return None

    append_token_span(
        input_ids,
        labels,
        tokenizer,
        f"{CHAT_START}{role}\n",
        train_on_span=False,
    )
    append_token_span(
        input_ids,
        labels,
        tokenizer,
        f"{content}{CHAT_END}\n",
        train_on_span=(role == "assistant"),
    )
    return role


def render_messages(messages: Any, tokenizer) -> tuple[list[int], list[int], bool]:
    if not isinstance(messages, list):
        return [], [], False

    input_ids: list[int] = []
    labels: list[int] = []
    ended_with_assistant = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = append_chat_turn(
            input_ids,
            labels,
            tokenizer,
            message.get("role", "user"),
            message.get("content", ""),
        )
        if role is not None:
            ended_with_assistant = role == "assistant"
    return input_ids, labels, ended_with_assistant


def format_sft_example(
    example: dict[str, Any],
    source: SFTSource,
    tokenizer,
) -> tuple[list[int], list[int], bool]:
    if source.messages_column and source.messages_column in example:
        return render_messages(example[source.messages_column], tokenizer)

    input_ids: list[int] = []
    labels: list[int] = []
    if source.system_column and source.system_column in example:
        system = normalize_text_value(example[source.system_column]).strip()
        if system:
            append_chat_turn(input_ids, labels, tokenizer, "system", system)

    user_parts = []
    for column in source.user_columns:
        if column in example:
            text = normalize_text_value(example[column]).strip()
            if text:
                user_parts.append(text)
    assistant = (
        normalize_text_value(example.get(source.assistant_column)).strip()
        if source.assistant_column
        else ""
    )
    if not user_parts or not assistant:
        return [], [], False

    append_chat_turn(input_ids, labels, tokenizer, "user", "\n\n".join(user_parts))
    append_chat_turn(input_ids, labels, tokenizer, "assistant", assistant)
    return input_ids, labels, True


def load_streaming_source(source: SFTSource, hf_token: str | None):
    load_kwargs = dict(source.load_kwargs)
    if hf_token is not None:
        load_kwargs["token"] = hf_token
    if source.config_name:
        ds = load_dataset(
            source.dataset_id,
            source.config_name,
            split=source.split,
            streaming=True,
            **load_kwargs,
        )
    else:
        ds = load_dataset(
            source.dataset_id,
            split=source.split,
            streaming=True,
            **load_kwargs,
        )
    if source.skip_examples > 0:
        ds = ds.skip(source.skip_examples)
    return ds


def flush_token_blocks(
    shard_blocks: list[list[int]],
    shard_labels: list[list[int]],
    tokenized_path: str,
    manifest: dict[str, Any],
) -> int:
    if not shard_blocks:
        return 0
    if len(shard_blocks) != len(shard_labels):
        raise ValueError(
            f"Mismatched token and label block counts: {len(shard_blocks)} != {len(shard_labels)}"
        )

    shards_root = os.path.join(tokenized_path, "shards")
    shard_name = f"shard_{len(manifest['shards']):05d}"
    shard_path = os.path.join(shards_root, shard_name)
    if os.path.exists(shard_path):
        shutil.rmtree(shard_path)

    ds = HFDataset.from_dict(
        {
            TOKEN_BLOCK_COLUMN: shard_blocks,
            LABEL_BLOCK_COLUMN: shard_labels,
        }
    )
    ds.save_to_disk(shard_path)

    shard_count = len(shard_blocks)
    manifest["total_blocks"] += shard_count
    manifest["total_training_tokens"] = manifest["total_blocks"] * manifest["context_length"]
    manifest["shards"].append(
        {
            "name": shard_name,
            "path": f"shards/{shard_name}",
            "blocks": shard_count,
            "training_tokens": shard_count * manifest["context_length"],
        }
    )
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)
    print(
        f"Saved {shard_count:,} token blocks to {shard_path} "
        f"({manifest['total_training_tokens']:,} training tokens total)"
    )
    return shard_count


def build_or_load_sft_tokenized_source(
    source: SFTSource | dict[str, Any],
    tokenizer_path: str,
    tokenizer,
    vocab_size: int,
    special_tokens: list[str],
    context_length_value: int,
    tokenized_path: str,
    hf_token: str | None = None,
    tokenized_shard_blocks: int = 8192,
):
    source = normalize_sft_source(source)
    label = sft_source_cache_label(source, vocab_size, context_length_value)
    if manifest_is_complete(tokenized_path, label):
        print(f"Loading tokenized SFT source '{label}' from {tokenized_path} ...")
        return load_tokenized_blocks(tokenized_path)

    eos_id = tokenizer.token_to_id(special_tokens[0])
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing special token {special_tokens[0]!r}")

    requested_blocks = max(1, math.ceil(source.token_budget / context_length_value))
    tokenizer_sha256 = file_sha256(tokenizer_path)

    print(
        f"Building tokenized SFT source '{label}' in {tokenized_path} "
        f"({requested_blocks:,} target blocks, "
        f"{requested_blocks * context_length_value:,} target training tokens) ..."
    )
    os.makedirs(tokenized_path, exist_ok=True)
    shards_root = os.path.join(tokenized_path, "shards")
    os.makedirs(shards_root, exist_ok=True)
    shutil.copy2(tokenizer_path, os.path.join(tokenized_path, "tokenizer.json"))

    manifest = {
        "completed": False,
        "pipeline_version": SFT_TOKENIZED_PIPELINE_VERSION,
        "label": label,
        "stage_name": STAGE_NAME,
        "source": source_to_manifest(source),
        "vocab_size": vocab_size,
        "context_length": context_length_value,
        "special_tokens": list(special_tokens),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_path": "tokenizer.json",
        "token_column": TOKEN_BLOCK_COLUMN,
        "label_column": LABEL_BLOCK_COLUMN,
        "ignore_index": IGNORE_INDEX,
        "requested_blocks": requested_blocks,
        "requested_training_tokens": requested_blocks * context_length_value,
        "docs_loaded": 0,
        "docs_used": 0,
        "empty_docs": 0,
        "total_blocks": 0,
        "total_training_tokens": 0,
        "shards": [],
    }
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)

    block_width = context_length_value + 1
    flat_ids: list[int] = []
    flat_labels: list[int] = []
    shard_blocks: list[list[int]] = []
    shard_labels: list[list[int]] = []
    docs_loaded = 0
    docs_used = 0
    empty_docs = 0

    ds = load_streaming_source(source, hf_token=hf_token)
    for example in ds:
        if manifest["total_blocks"] + len(shard_blocks) >= requested_blocks:
            break

        docs_loaded += 1
        input_ids, labels, ended_with_assistant = format_sft_example(
            example,
            source,
            tokenizer,
        )
        if not input_ids:
            empty_docs += 1
            continue
        if len(input_ids) != len(labels):
            raise ValueError(
                f"Token/label length mismatch for source {source.label!r}: "
                f"{len(input_ids)} != {len(labels)}"
            )

        docs_used += 1
        flat_ids.extend(input_ids)
        flat_labels.extend(labels)
        flat_ids.append(eos_id)
        flat_labels.append(eos_id if ended_with_assistant else IGNORE_INDEX)

        while (
            len(flat_ids) >= block_width
            and manifest["total_blocks"] + len(shard_blocks) < requested_blocks
        ):
            block_ids = flat_ids[:block_width]
            block_labels = flat_labels[:block_width]
            if any(label != IGNORE_INDEX for label in block_labels[1:]):
                shard_blocks.append(block_ids)
                shard_labels.append(block_labels)
            del flat_ids[:block_width]
            del flat_labels[:block_width]
            if len(shard_blocks) >= tokenized_shard_blocks:
                manifest["docs_loaded"] = docs_loaded
                manifest["docs_used"] = docs_used
                manifest["empty_docs"] = empty_docs
                flush_token_blocks(shard_blocks, shard_labels, tokenized_path, manifest)
                shard_blocks = []
                shard_labels = []
                gc.collect()

        if docs_loaded % 10_000 == 0:
            print(
                f"  {source.label}: loaded {docs_loaded:,} docs, "
                f"saved {manifest['total_blocks']:,} blocks, "
                f"buffered {len(shard_blocks):,} blocks"
            )

    if shard_blocks:
        manifest["docs_loaded"] = docs_loaded
        manifest["docs_used"] = docs_used
        manifest["empty_docs"] = empty_docs
        flush_token_blocks(shard_blocks, shard_labels, tokenized_path, manifest)

    if manifest["total_blocks"] < requested_blocks:
        print(
            f"SFT source '{source.label}' exhausted at "
            f"{manifest['total_training_tokens']:,} tokens before requested "
            f"{requested_blocks * context_length_value:,}"
        )
    if manifest["total_blocks"] == 0:
        raise ValueError(
            f"SFT source {source.label!r} produced zero token blocks. "
            "Check its column mapping or use a shorter context length."
        )

    manifest["completed"] = True
    manifest["docs_loaded"] = docs_loaded
    manifest["docs_used"] = docs_used
    manifest["empty_docs"] = empty_docs
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)
    with open(os.path.join(tokenized_path, "label.txt"), "w", encoding="utf-8") as f:
        f.write(label + "\n")

    print(f"Tokenized SFT source label: {label}")
    return load_tokenized_blocks(tokenized_path)


def build_or_load_mixed_sft_tokenized_blocks(
    sources: list[SFTSource | dict[str, Any]],
    tokenizer_path: str,
    tokenizer,
    vocab_size: int,
    special_tokens: list[str],
    context_length_value: int,
    tokenized_root: str,
    data_label: str | None = None,
    hf_token: str | None = None,
    tokenized_shard_blocks: int = 8192,
):
    normalized_sources = [normalize_sft_source(source) for source in sources]
    if not normalized_sources:
        raise ValueError("At least one SFT source is required")

    mix_label = (
        sanitize_label(data_label)
        if data_label
        else default_sft_mix_label(normalized_sources, vocab_size, context_length_value)
    )
    mix_root = os.path.join(tokenized_root, mix_label)
    sources_root = os.path.join(mix_root, "sources")
    os.makedirs(sources_root, exist_ok=True)

    tokenized_sources = []
    source_summaries = []
    total_blocks = 0
    for source in normalized_sources:
        source_label = sft_source_cache_label(source, vocab_size, context_length_value)
        source_path = os.path.join(sources_root, source_label)
        token_blocks, source_manifest = build_or_load_sft_tokenized_source(
            source=source,
            tokenizer_path=tokenizer_path,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            context_length_value=context_length_value,
            tokenized_path=source_path,
            hf_token=hf_token,
            tokenized_shard_blocks=tokenized_shard_blocks,
        )
        tokenized_sources.append(token_blocks)
        source_blocks = int(source_manifest["total_blocks"])
        total_blocks += source_blocks
        source_summaries.append(
            {
                "label": source.label,
                "cache_label": source_manifest["label"],
                "dataset_id": source.dataset_id,
                "config_name": source.config_name,
                "split": source.split,
                "skip_examples": source.skip_examples,
                "token_budget": source.token_budget,
                "blocks": source_blocks,
                "training_tokens": int(source_manifest["total_training_tokens"]),
                "docs_loaded": int(source_manifest["docs_loaded"]),
                "docs_used": int(source_manifest["docs_used"]),
                "empty_docs": int(source_manifest["empty_docs"]),
                "shards": len(source_manifest["shards"]),
                "manifest_path": os.path.join(source_path, "manifest.json"),
            }
        )

    if total_blocks < 2:
        raise ValueError("Need at least two SFT token blocks to create train and val splits")

    mixed_blocks = (
        tokenized_sources[0]
        if len(tokenized_sources) == 1
        else concatenate_datasets(tokenized_sources)
    )
    mixed_blocks = mixed_blocks.with_format("torch")

    manifest = {
        "completed": True,
        "pipeline_version": SFT_TOKENIZED_PIPELINE_VERSION,
        "label": mix_label,
        "stage_name": STAGE_NAME,
        "vocab_size": vocab_size,
        "context_length": context_length_value,
        "special_tokens": list(special_tokens),
        "token_column": TOKEN_BLOCK_COLUMN,
        "label_column": LABEL_BLOCK_COLUMN,
        "ignore_index": IGNORE_INDEX,
        "tokenized_root": tokenized_root,
        "total_blocks": len(mixed_blocks),
        "total_training_tokens": len(mixed_blocks) * context_length_value,
        "sources": source_summaries,
    }
    write_json_atomic(os.path.join(mix_root, "manifest.json"), manifest)
    with open(os.path.join(mix_root, "label.txt"), "w", encoding="utf-8") as f:
        f.write(mix_label + "\n")

    print(
        f"Created mixed SFT dataset '{mix_label}' at {mix_root}: "
        f"{len(mixed_blocks):,} blocks, "
        f"{len(mixed_blocks) * context_length_value:,} training tokens, "
        f"{len(source_summaries):,} sources"
    )
    return mixed_blocks, manifest


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
    tokenizer_path = f"tokenizer/tokenizer_{VOCAB_SIZE}.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"SFT expects an existing tokenizer at {tokenizer_path}. Run pretraining/tokenizer setup first."
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")

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

    block_dataset = HFSFTMaskedDataset(token_blocks, context_length)
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
        tokens_seen = int(checkpoint.get(f"{STAGE_NAME}_tokens_seen", checkpoint.get("sft_tokens_seen", 0)))
        best_val_ppl = checkpoint.get("best_val_ppl", best_val_ppl)
        no_improve = int(checkpoint.get("no_improve", 0))
        print(f"Resuming SFT state at step {step:,}, tokens_seen={tokens_seen:,}")
    else:
        print("Loaded model weights only; SFT optimizer and LR schedule start fresh")

    start_sample = min(tokens_seen // context_length, len(train_dataset))
    if start_sample > 0:
        print(f"Skipping {start_sample:,} already-seen SFT training samples")
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
            "val_tokens": val_samples * context_length,
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
