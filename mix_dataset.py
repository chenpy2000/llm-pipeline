from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, hf_hub_download

from data_pipeline import (
    TOKENIZED_PIPELINE_VERSION,
    TOKEN_BLOCK_COLUMN,
    file_sha256,
    fineweb_parquet_prefix,
    get_worker_tokenizer,
    load_tokenized_blocks,
    manifest_is_complete,
    sanitize_label,
    tokenize_text_batch,
    write_json_atomic,
)


CPT_TOKENIZED_PIPELINE_VERSION = TOKENIZED_PIPELINE_VERSION
DEFAULT_TEXT_COLUMNS = (
    "text",
    "content",
    "code",
    "body",
    "message",
    "diff",
    "title",
    "description",
)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
SOFTWARE_HERITAGE_CONTENT_URL = "https://softwareheritage.s3.amazonaws.com/content/{blob_id}"


@dataclass(frozen=True)
class CPTSource:
    label: str
    dataset_id: str
    token_budget: int
    split: str = "train"
    config_name: str | None = None
    data_dir: str | None = None
    text_column: str | None = None
    text_columns: tuple[str, ...] = ()
    blob_id_column: str | None = None
    fim_rate: float = 0.0
    num_docs: int | None = None
    source_files: tuple[str, ...] = ()
    file_prefix: str | None = None
    file_format: str | None = None
    load_kwargs: dict[str, Any] = field(default_factory=dict)


def normalize_cpt_source(source: CPTSource | dict[str, Any]) -> CPTSource:
    if isinstance(source, CPTSource):
        return source

    source = dict(source)
    text_columns = source.get("text_columns") or ()
    if isinstance(text_columns, str):
        text_columns = (text_columns,)
    else:
        text_columns = tuple(text_columns)

    source_files = (
        source.get("source_files")
        or source.get("data_files")
        or source.get("parquet_files")
        or ()
    )
    if isinstance(source_files, str):
        source_files = (source_files,)
    else:
        source_files = tuple(source_files)

    return CPTSource(
        label=source["label"],
        dataset_id=source["dataset_id"],
        token_budget=int(source["token_budget"]),
        split=source.get("split", "train"),
        config_name=source.get("config_name"),
        data_dir=source.get("data_dir"),
        text_column=source.get("text_column"),
        text_columns=text_columns,
        blob_id_column=source.get("blob_id_column"),
        fim_rate=float(source.get("fim_rate", 0.0)),
        num_docs=(
            int(source["num_docs"])
            if source.get("num_docs") is not None
            else None
        ),
        source_files=source_files,
        file_prefix=source.get("file_prefix") or source.get("parquet_prefix"),
        file_format=source.get("file_format"),
        load_kwargs=dict(source.get("load_kwargs") or {}),
    )


def source_to_manifest(source: CPTSource) -> dict[str, Any]:
    return {
        "label": source.label,
        "dataset_id": source.dataset_id,
        "config_name": source.config_name,
        "data_dir": source.data_dir,
        "split": source.split,
        "text_column": source.text_column,
        "text_columns": list(source.text_columns),
        "blob_id_column": source.blob_id_column,
        "token_budget": source.token_budget,
        "fim_rate": source.fim_rate,
        "num_docs": source.num_docs,
        "source_files": list(source.source_files),
        "file_prefix": source.file_prefix,
        "file_format": source.file_format,
        "load_kwargs": dict(source.load_kwargs),
    }


def default_cpt_mix_label(
    sources: list[CPTSource],
    vocab_size: int,
    context_length: int,
) -> str:
    digest_input = json.dumps(
        [source_to_manifest(source) for source in sources],
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:12]
    return sanitize_label(f"cpt_mix_{digest}_ctx{context_length}_v{vocab_size}")


def cpt_source_cache_label(source: CPTSource, vocab_size: int, context_length: int) -> str:
    digest = hashlib.sha1(
        json.dumps(source_to_manifest(source), sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return sanitize_label(
        f"{source.label}_{digest}_ctx{context_length}_v{vocab_size}"
    )


def download_software_heritage_text(blob_id: str, timeout: int = 60) -> str:
    url = SOFTWARE_HERITAGE_CONTENT_URL.format(blob_id=blob_id)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return ""
        raise

    try:
        payload = gzip.decompress(payload)
    except OSError:
        pass
    return payload.decode("utf-8", errors="ignore")


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


def extract_text(example: dict[str, Any], source: CPTSource) -> str:
    if source.text_column and source.text_column in example:
        text = normalize_text_value(example[source.text_column])
        if text:
            return text

    parts = []
    for column in source.text_columns:
        if column in example:
            text = normalize_text_value(example[column])
            if text:
                parts.append(text)
    if parts:
        return "\n\n".join(parts)

    if source.blob_id_column and source.blob_id_column in example:
        blob_id = normalize_text_value(example[source.blob_id_column]).strip()
        if blob_id:
            return download_software_heritage_text(blob_id)

    for column in DEFAULT_TEXT_COLUMNS:
        if column in example:
            text = normalize_text_value(example[column])
            if text:
                return text

    return ""


def should_apply_fim(fim_rate: float, doc_index: int) -> bool:
    if fim_rate <= 0:
        return False
    threshold = max(0, min(1000, int(round(fim_rate * 1000))))
    return doc_index % 1000 < threshold


def to_fim_text(text: str) -> str:
    if len(text) < 120:
        return text
    first = len(text) // 3
    second = (2 * len(text)) // 3
    if first >= second:
        return text

    prefix = text[:first]
    middle = text[first:second]
    suffix = text[second:]
    return f"{FIM_PREFIX}\n{prefix}\n{FIM_SUFFIX}\n{suffix}\n{FIM_MIDDLE}\n{middle}"


def source_file_format(source_file: str) -> str:
    lower = source_file.lower()
    if lower.endswith(".parquet"):
        return "parquet"
    if (
        lower.endswith(".json")
        or lower.endswith(".jsonl")
        or lower.endswith(".json.gz")
        or lower.endswith(".jsonl.gz")
    ):
        return "json"
    raise ValueError(f"Unsupported source shard format for {source_file!r}")


def format_extensions(file_format: str) -> tuple[str, ...]:
    if file_format == "parquet":
        return (".parquet",)
    if file_format == "json":
        return (".json", ".jsonl", ".json.gz", ".jsonl.gz")
    raise ValueError(f"Unsupported file_format {file_format!r}")


def ensure_prefix(value: str) -> str:
    return value.rstrip("/") + "/"


def candidate_prefixes(source: CPTSource) -> list[str]:
    prefixes = []
    if source.file_prefix:
        prefixes.append(ensure_prefix(source.file_prefix))
    if source.data_dir:
        prefixes.append(ensure_prefix(source.data_dir))
    if source.config_name and source.config_name != "default":
        prefixes.append(ensure_prefix(source.config_name))
        prefixes.append(ensure_prefix(f"data/{source.config_name}"))
        prefixes.append(ensure_prefix(fineweb_parquet_prefix(source.config_name)))
    if source.config_name in (None, "default"):
        prefixes.append("data/")

    seen = set()
    unique = []
    for prefix in prefixes:
        if prefix not in seen:
            seen.add(prefix)
            unique.append(prefix)
    return unique


def filter_files_by_split(files: list[str], split: str) -> list[str]:
    split = split.lower()
    matches = []
    for filename in files:
        lower = filename.lower()
        basename = Path(lower).name
        if basename.startswith(f"{split}-") or f"/{split}-" in lower:
            matches.append(filename)
    return matches or files


def list_source_files(source: CPTSource, hf_token: str | None) -> list[str]:
    if source.source_files:
        return list(source.source_files)

    revision = source.load_kwargs.get("revision")
    api = HfApi(token=hf_token) if hf_token else HfApi()
    repo_files = api.list_repo_files(
        source.dataset_id,
        repo_type="dataset",
        revision=revision,
    )
    file_formats = [source.file_format] if source.file_format else ["parquet", "json"]
    prefixes = candidate_prefixes(source)

    for file_format in file_formats:
        extensions = format_extensions(file_format)
        format_files = [
            filename
            for filename in repo_files
            if filename.lower().endswith(extensions)
        ]
        if not format_files:
            continue

        for prefix in prefixes:
            prefixed = [
                filename
                for filename in format_files
                if filename.startswith(prefix)
            ]
            prefixed = filter_files_by_split(prefixed, source.split)
            if prefixed:
                return sorted(prefixed)

        split_files = filter_files_by_split(format_files, source.split)
        if split_files:
            return sorted(split_files)

    raise ValueError(
        f"No supported raw shards found for {source.dataset_id!r} "
        f"(config={source.config_name!r}, data_dir={source.data_dir!r})"
    )


def download_raw_source_file(
    source: CPTSource,
    source_file: str,
    raw_data_dir: str,
    hf_token: str | None,
) -> str:
    os.makedirs(raw_data_dir, exist_ok=True)
    print(f"Downloading raw dataset shard {source_file} ...")
    download_kwargs = {
        "repo_id": source.dataset_id,
        "filename": source_file,
        "repo_type": "dataset",
        "local_dir": raw_data_dir,
        "force_download": True,
    }
    if hf_token is not None:
        download_kwargs["token"] = hf_token
    if source.load_kwargs.get("revision") is not None:
        download_kwargs["revision"] = source.load_kwargs["revision"]
    return hf_hub_download(**download_kwargs)


def cleanup_raw_source_shard(local_file: str | None, cache_dir: str | None) -> None:
    try:
        if local_file and os.path.exists(local_file):
            os.remove(local_file)
            print(f"Deleted raw dataset shard {local_file}")
    finally:
        if cache_dir and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)


def load_raw_source_dataset(
    local_file: str,
    file_format: str,
    split: str,
    cache_dir: str,
):
    return load_dataset(
        file_format,
        data_files={split: local_file},
        split=split,
        cache_dir=cache_dir,
    )


def can_use_pretrain_tokenizer_batch(source: CPTSource, column_names: list[str]) -> bool:
    return (
        source.text_column is not None
        and source.text_column in column_names
        and not source.text_columns
        and source.blob_id_column is None
        and source.fim_rate <= 0
    )


def validate_source_columns(source: CPTSource, column_names: list[str]) -> None:
    candidates = []
    if source.text_column:
        candidates.append(source.text_column)
    candidates.extend(source.text_columns)
    if source.blob_id_column:
        candidates.append(source.blob_id_column)
    candidates.extend(DEFAULT_TEXT_COLUMNS)

    if any(column in column_names for column in candidates):
        return

    raise ValueError(
        f"Could not find a usable text/blob column for source {source.label!r}; "
        f"available columns are {column_names}"
    )


def tokenize_cpt_text_batch(
    batch,
    indices,
    tokenizer_path,
    eos_id,
    context_length,
    source,
):
    tokenizer = get_worker_tokenizer(tokenizer_path)
    batch_size = len(indices)
    texts = []
    for row_index in range(batch_size):
        example = {
            column: values[row_index]
            for column, values in batch.items()
        }
        text = extract_text(example, source)
        if not text:
            continue
        if should_apply_fim(source.fim_rate, indices[row_index]):
            text = to_fim_text(text)
        texts.append(text)

    if not texts:
        return {TOKEN_BLOCK_COLUMN: []}

    encodings = tokenizer.encode_batch_fast(texts)
    flat_ids = []
    for encoding in encodings:
        flat_ids.extend(encoding.ids)
        flat_ids.append(eos_id)

    block_width = context_length + 1
    usable_tokens = (len(flat_ids) // block_width) * block_width
    blocks = [
        flat_ids[start : start + block_width]
        for start in range(0, usable_tokens, block_width)
    ]
    return {TOKEN_BLOCK_COLUMN: blocks}


def build_or_load_cpt_tokenized_source(
    source: CPTSource | dict[str, Any],
    tokenizer_path: str,
    tokenizer,
    vocab_size: int,
    special_tokens: list[str],
    context_length: int,
    tokenized_path: str,
    raw_data_dir: str,
    hf_token: str | None = None,
    encode_workers: int = 1,
    tokenize_batch_size: int = 500,
):
    source = normalize_cpt_source(source)
    label = cpt_source_cache_label(source, vocab_size, context_length)
    if manifest_is_complete(tokenized_path, label):
        print(f"Loading tokenized CPT source '{label}' from {tokenized_path} ...")
        return load_tokenized_blocks(tokenized_path)

    eos_id = tokenizer.token_to_id(special_tokens[0])
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing special token {special_tokens[0]!r}")

    requested_blocks = max(1, math.ceil(source.token_budget / context_length))
    tokenizer_sha256 = file_sha256(tokenizer_path)
    source_files = list_source_files(source, hf_token=hf_token)

    print(
        f"Building tokenized CPT source '{label}' in {tokenized_path} "
        f"({requested_blocks:,} target blocks, "
        f"{requested_blocks * context_length:,} target training tokens) ..."
    )
    os.makedirs(tokenized_path, exist_ok=True)
    shards_root = os.path.join(tokenized_path, "shards")
    os.makedirs(shards_root, exist_ok=True)
    shutil.copy2(tokenizer_path, os.path.join(tokenized_path, "tokenizer.json"))

    manifest = {
        "completed": False,
        "pipeline_version": CPT_TOKENIZED_PIPELINE_VERSION,
        "label": label,
        "source": source_to_manifest(source),
        "source_files": source_files,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "special_tokens": list(special_tokens),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_path": "tokenizer.json",
        "token_column": TOKEN_BLOCK_COLUMN,
        "requested_blocks": requested_blocks,
        "requested_training_tokens": requested_blocks * context_length,
        "docs_loaded": 0,
        "total_blocks": 0,
        "total_training_tokens": 0,
        "shards": [],
    }
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)

    docs_seen = 0
    total_blocks = 0
    for source_index, source_file in enumerate(source_files):
        if total_blocks >= requested_blocks:
            break
        if source.num_docs is not None and docs_seen >= source.num_docs:
            break

        local_file = None
        ds = None
        ds_tok = None
        cache_dir = os.path.join(
            raw_data_dir,
            "_datasets_cache",
            label,
            f"tokenize_{source_index:05d}",
        )
        try:
            local_file = download_raw_source_file(
                source=source,
                source_file=source_file,
                raw_data_dir=raw_data_dir,
                hf_token=hf_token,
            )
            file_format = source.file_format or source_file_format(source_file)
            ds = load_raw_source_dataset(
                local_file=local_file,
                file_format=file_format,
                split=source.split,
                cache_dir=cache_dir,
            )
            validate_source_columns(source, ds.column_names)

            source_docs = len(ds)
            docs_in_shard = source_docs
            if source.num_docs is not None:
                remaining_docs = source.num_docs - docs_seen
                docs_in_shard = min(remaining_docs, source_docs)
            if docs_in_shard < source_docs:
                ds = ds.select(range(docs_in_shard))
            docs_seen += len(ds)

            if len(ds) == 0:
                continue

            num_proc = max(1, min(encode_workers, len(ds)))
            map_num_proc = num_proc if num_proc > 1 else None
            print(
                f"Tokenizing {len(ds):,} docs from {source_file} "
                f"with {num_proc} worker(s) ..."
            )
            if can_use_pretrain_tokenizer_batch(source, ds.column_names):
                ds_tok = ds.map(
                    tokenize_text_batch,
                    batched=True,
                    batch_size=tokenize_batch_size,
                    num_proc=map_num_proc,
                    remove_columns=ds.column_names,
                    fn_kwargs={
                        "tokenizer_path": tokenizer_path,
                        "eos_id": eos_id,
                        "context_length": context_length,
                        "text_column": source.text_column,
                    },
                    desc=f"Tokenizing {Path(source_file).name}",
                )
            else:
                ds_tok = ds.map(
                    tokenize_cpt_text_batch,
                    batched=True,
                    with_indices=True,
                    batch_size=tokenize_batch_size,
                    num_proc=map_num_proc,
                    remove_columns=ds.column_names,
                    fn_kwargs={
                        "tokenizer_path": tokenizer_path,
                        "eos_id": eos_id,
                        "context_length": context_length,
                        "source": source,
                    },
                    desc=f"Tokenizing {Path(source_file).name}",
                )

            remaining_blocks = requested_blocks - total_blocks
            if len(ds_tok) > remaining_blocks:
                ds_tok = ds_tok.select(range(remaining_blocks))

            shard_blocks = len(ds_tok)
            if shard_blocks == 0:
                continue

            shard_name = f"shard_{len(manifest['shards']):05d}"
            shard_path = os.path.join(shards_root, shard_name)
            if os.path.exists(shard_path):
                shutil.rmtree(shard_path)
            ds_tok.save_to_disk(shard_path)

            total_blocks += shard_blocks
            manifest["docs_loaded"] = docs_seen
            manifest["total_blocks"] = total_blocks
            manifest["total_training_tokens"] = total_blocks * context_length
            manifest["shards"].append(
                {
                    "name": shard_name,
                    "path": f"shards/{shard_name}",
                    "source_file": source_file,
                    "docs": len(ds),
                    "blocks": shard_blocks,
                    "training_tokens": shard_blocks * context_length,
                }
            )
            write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)
            print(
                f"Saved {shard_blocks:,} token blocks to {shard_path} "
                f"({manifest['total_training_tokens']:,} training tokens total)"
            )
        finally:
            ds_tok = None
            ds = None
            gc.collect()
            cleanup_raw_source_shard(local_file, cache_dir)

    if source.num_docs is not None and docs_seen < source.num_docs:
        print(f"Only found {docs_seen:,} docs for source '{source.label}'")
    if total_blocks < requested_blocks:
        print(
            f"CPT source '{source.label}' exhausted at "
            f"{total_blocks * context_length:,} tokens before requested "
            f"{requested_blocks * context_length:,}"
        )
    if total_blocks == 0:
        raise ValueError(
            f"CPT source {source.label!r} produced zero token blocks. "
            "Use more documents or a shorter context length."
        )

    manifest["completed"] = True
    manifest["docs_loaded"] = docs_seen
    manifest["total_blocks"] = total_blocks
    manifest["total_training_tokens"] = total_blocks * context_length
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)
    with open(os.path.join(tokenized_path, "label.txt"), "w", encoding="utf-8") as f:
        f.write(label + "\n")

    print(f"Tokenized CPT source label: {label}")
    return load_tokenized_blocks(tokenized_path)


def build_or_load_mixed_tokenized_blocks(
    sources: list[CPTSource | dict[str, Any]],
    tokenizer_path: str,
    tokenizer,
    vocab_size: int,
    special_tokens: list[str],
    context_length: int,
    tokenized_root: str,
    raw_data_dir: str,
    data_label: str | None = None,
    hf_token: str | None = None,
    encode_workers: int = 1,
    tokenize_batch_size: int = 500,
):
    normalized_sources = [normalize_cpt_source(source) for source in sources]
    if not normalized_sources:
        raise ValueError("At least one CPT source is required")

    mix_label = (
        sanitize_label(data_label)
        if data_label
        else default_cpt_mix_label(normalized_sources, vocab_size, context_length)
    )
    mix_root = os.path.join(tokenized_root, mix_label)
    sources_root = os.path.join(mix_root, "sources")
    os.makedirs(sources_root, exist_ok=True)

    tokenized_sources = []
    source_summaries = []
    total_blocks = 0
    for source in normalized_sources:
        source_label = cpt_source_cache_label(source, vocab_size, context_length)
        source_path = os.path.join(sources_root, source_label)
        token_blocks, source_manifest = build_or_load_cpt_tokenized_source(
            source=source,
            tokenizer_path=tokenizer_path,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            context_length=context_length,
            tokenized_path=source_path,
            raw_data_dir=raw_data_dir,
            hf_token=hf_token,
            encode_workers=encode_workers,
            tokenize_batch_size=tokenize_batch_size,
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
                "data_dir": source.data_dir,
                "split": source.split,
                "text_column": source.text_column,
                "text_columns": list(source.text_columns),
                "blob_id_column": source.blob_id_column,
                "token_budget": source.token_budget,
                "blocks": source_blocks,
                "training_tokens": int(source_manifest["total_training_tokens"]),
                "docs_loaded": int(source_manifest["docs_loaded"]),
                "shards": len(source_manifest["shards"]),
                "manifest_path": os.path.join(source_path, "manifest.json"),
            }
        )

    if total_blocks < 2:
        raise ValueError("Need at least two CPT token blocks to create train and val splits")

    mixed_blocks = (
        tokenized_sources[0]
        if len(tokenized_sources) == 1
        else concatenate_datasets(tokenized_sources)
    )
    mixed_blocks = mixed_blocks.with_format("torch")

    manifest = {
        "completed": True,
        "pipeline_version": CPT_TOKENIZED_PIPELINE_VERSION,
        "label": mix_label,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "special_tokens": list(special_tokens),
        "tokenized_root": tokenized_root,
        "raw_data_dir": raw_data_dir,
        "total_blocks": len(mixed_blocks),
        "total_training_tokens": len(mixed_blocks) * context_length,
        "sources": source_summaries,
    }
    write_json_atomic(os.path.join(mix_root, "manifest.json"), manifest)
    with open(os.path.join(mix_root, "label.txt"), "w", encoding="utf-8") as f:
        f.write(mix_label + "\n")

    print(
        f"Created mixed CPT dataset '{mix_label}' at {mix_root}: "
        f"{len(mixed_blocks):,} blocks, "
        f"{len(mixed_blocks) * context_length:,} training tokens, "
        f"{len(source_summaries):,} sources"
    )
    return mixed_blocks, manifest
