from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import torch
from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Dataset, Subset

from mix_dataset import (
    TOKEN_BLOCK_COLUMN,
    build_or_load_mixed_tokenized_blocks,
    file_sha256,
    load_tokenized_blocks,
    manifest_is_complete,
    sanitize_label,
    write_json_atomic,
)


class LMDataset(Dataset):
    """
    Chunks a flat token stream into (input, target) pairs for
    next-token prediction.

    input:  tokens[i   : i + block_size]
    target: tokens[i+1 : i + block_size + 1]
    """

    def __init__(self, token_ids, block_size):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return (len(self.token_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.token_ids[start     : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


class HFCausalLMDataset(Dataset):
    """Wrap fixed-width token blocks from a Hugging Face Dataset."""

    def __init__(self, hf_dataset, block_size, column="input_ids"):
        self.hf_dataset = hf_dataset
        self.block_size = block_size
        self.column = column

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        token_ids = self.hf_dataset[idx][self.column]
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)

        expected_width = self.block_size + 1
        if token_ids.numel() != expected_width:
            raise ValueError(
                f"Expected token block width {expected_width}, got {token_ids.numel()}"
            )

        return token_ids[:-1], token_ids[1:]


@dataclass
class TrainingData:
    block_dataset: Any
    train_dataset: Subset
    val_dataset: Subset
    val_loader: DataLoader
    data_manifest: dict[str, Any]
    total_tokens: int
    val_samples: int
    batch_size: int
    loader_kwargs: dict[str, Any]

    def make_train_loader(self, start_sample: int = 0, shuffle: bool = False) -> DataLoader:
        start_sample = max(0, min(int(start_sample), len(self.train_dataset)))
        train_subset = Subset(
            self.train_dataset,
            range(start_sample, len(self.train_dataset)),
        )
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            **self.loader_kwargs,
        )


def dataloader_kwargs(device, num_workers: int) -> dict[str, Any]:
    device_type = getattr(device, "type", str(device))
    return {
        "num_workers": num_workers,
        "pin_memory": device_type == "cuda",
        "persistent_workers": num_workers > 0,
    }


def split_block_dataset(block_dataset, context_length: int, val_tokens: int):
    if len(block_dataset) < 2:
        raise ValueError("Need at least two token blocks to create train and val splits")

    val_samples = max(1, val_tokens // context_length)
    val_samples = min(val_samples, max(1, len(block_dataset) // 10))
    val_samples = min(val_samples, len(block_dataset) - 1)
    split = len(block_dataset) - val_samples
    train_dataset = Subset(block_dataset, range(split))
    val_dataset = Subset(block_dataset, range(split, len(block_dataset)))
    return train_dataset, val_dataset, val_samples


def build_training_data(
    token_blocks,
    data_manifest: dict[str, Any],
    dataset_cls,
    context_length: int,
    val_tokens: int,
    batch_size: int,
    device,
    num_workers: int,
) -> TrainingData:
    block_dataset = dataset_cls(token_blocks, context_length)
    train_dataset, val_dataset, val_samples = split_block_dataset(
        block_dataset,
        context_length=context_length,
        val_tokens=val_tokens,
    )
    loader_kwargs = dataloader_kwargs(device, num_workers)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return TrainingData(
        block_dataset=block_dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_loader=val_loader,
        data_manifest=data_manifest,
        total_tokens=int(data_manifest["total_training_tokens"]),
        val_samples=val_samples,
        batch_size=batch_size,
        loader_kwargs=loader_kwargs,
    )


def build_causal_training_data(
    sources: list[dict[str, Any]],
    tokenizer_path: str,
    tokenizer,
    vocab_size: int,
    special_tokens: list[str],
    context_length: int,
    tokenized_root: str,
    raw_data_dir: str,
    data_label: str | None,
    val_tokens: int,
    batch_size: int,
    device,
    num_workers: int,
    hf_token: str | None = None,
    encode_workers: int = 1,
    tokenize_batch_size: int = 500,
) -> TrainingData:
    token_blocks, data_manifest = build_or_load_mixed_tokenized_blocks(
        sources=sources,
        tokenizer_path=tokenizer_path,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        context_length=context_length,
        tokenized_root=tokenized_root,
        raw_data_dir=raw_data_dir,
        data_label=data_label,
        hf_token=hf_token,
        encode_workers=encode_workers,
        tokenize_batch_size=tokenize_batch_size,
    )
    return build_training_data(
        token_blocks=token_blocks,
        data_manifest=data_manifest,
        dataset_cls=HFCausalLMDataset,
        context_length=context_length,
        val_tokens=val_tokens,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )


STAGE_NAME = "sft_qa_cot"


def set_sft_stage_name(stage_name: str) -> None:
    global STAGE_NAME
    STAGE_NAME = stage_name


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
