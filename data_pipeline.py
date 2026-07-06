import gc
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi, hf_hub_download

from tokenizer import Tokenizer


TOKENIZED_PIPELINE_VERSION = 1
TOKEN_BLOCK_COLUMN = "input_ids"
_TOKENIZER_CACHE = {}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def sanitize_label(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value)
    value = value.strip(".-")
    return value or "dataset"


def fineweb_parquet_prefix(config_name):
    sample_match = re.fullmatch(r"sample-(.+)", config_name)
    if sample_match:
        return f"sample/{sample_match.group(1)}"
    return f"data/{config_name}"


def list_parquet_files(dataset_id, config_name):
    prefix = fineweb_parquet_prefix(config_name).rstrip("/") + "/"
    files = HfApi().list_repo_files(dataset_id, repo_type="dataset")
    parquet_files = sorted(
        filename
        for filename in files
        if filename.startswith(prefix) and filename.endswith(".parquet")
    )
    if not parquet_files:
        raise ValueError(
            f"No Parquet files found for {dataset_id!r} under {prefix!r}"
        )
    return parquet_files


def default_data_label(
    dataset_id,
    config_name,
    split,
    text_column,
    num_docs,
    vocab_size,
    context_length,
    tokenizer_sha256,
    special_tokens,
):
    label_data = {
        "pipeline_version": TOKENIZED_PIPELINE_VERSION,
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "text_column": text_column,
        "num_docs": num_docs,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "tokenizer_sha256": tokenizer_sha256,
        "special_tokens": list(special_tokens),
    }
    short_hash = hashlib.sha256(
        json.dumps(label_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    dataset_name = dataset_id.rsplit("/", 1)[-1]
    return sanitize_label(
        f"{dataset_name}_{config_name}_{split}_docs{num_docs}_"
        f"ctx{context_length}_v{vocab_size}_{short_hash}"
    )


def cached_tokenizer_path(tokenized_root, data_label):
    if not data_label:
        return None
    path = os.path.join(tokenized_root, data_label, "tokenizer.json")
    return path if os.path.exists(path) else None


def load_or_train_tokenizer(
    tokenizer_path,
    vocab_size,
    special_tokens,
    dataset_id,
    config_name,
    text_column,
    num_docs,
    raw_parquet_dir,
    tokenized_root,
    data_label=None,
    iterator_batch_size=1000,
):
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")
        if tokenizer.loaded_from_legacy:
            tokenizer.save(tokenizer_path)
            print(f"Migrated legacy tokenizer to HF format at {tokenizer_path}")
        return tokenizer

    cached_path = cached_tokenizer_path(tokenized_root, data_label)
    if cached_path is not None:
        tokenizer = Tokenizer.from_file(cached_path)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Restored tokenizer from tokenized cache {cached_path}")
        return tokenizer

    print("No saved tokenizer found, training a new one from raw Parquet shards ...")
    parquet_files = list_parquet_files(dataset_id, config_name)
    tokenizer = Tokenizer()
    trainer = Tokenizer.build_bpe_trainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    tokenizer.train_from_iterator(
        iter_downloaded_texts(
            parquet_files=parquet_files,
            dataset_id=dataset_id,
            raw_parquet_dir=raw_parquet_dir,
            text_column=text_column,
            max_docs=num_docs,
            batch_size=iterator_batch_size,
        ),
        trainer=trainer,
        length=num_docs,
    )
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path} (vocab size: {tokenizer.vocab_size})")
    return tokenizer


def iter_downloaded_texts(
    parquet_files,
    dataset_id,
    raw_parquet_dir,
    text_column,
    max_docs,
    batch_size=1000,
):
    docs_seen = 0
    for shard_index, source_file in enumerate(parquet_files):
        if docs_seen >= max_docs:
            break

        local_parquet = download_raw_parquet(dataset_id, source_file, raw_parquet_dir)
        cache_dir = os.path.join(raw_parquet_dir, "_datasets_cache", f"tokenizer_{shard_index:05d}")
        ds = None
        try:
            ds = load_dataset(
                "parquet",
                data_files=local_parquet,
                split="train",
                cache_dir=cache_dir,
            )
            if text_column not in ds.column_names:
                raise ValueError(
                    f"Expected column {text_column!r} in {source_file}, got {ds.column_names}"
                )

            remaining_docs = max_docs - docs_seen
            if remaining_docs < len(ds):
                ds = ds.select(range(remaining_docs))
            docs_seen += len(ds)

            for batch in ds.iter(batch_size=batch_size):
                for text in batch[text_column]:
                    yield text
        finally:
            del ds
            gc.collect()
            cleanup_raw_shard(local_parquet, cache_dir)


def download_raw_parquet(dataset_id, source_file, raw_parquet_dir):
    os.makedirs(raw_parquet_dir, exist_ok=True)
    print(f"Downloading raw Parquet shard {source_file} ...")
    return hf_hub_download(
        repo_id=dataset_id,
        filename=source_file,
        repo_type="dataset",
        local_dir=raw_parquet_dir,
        force_download=True,
    )


def cleanup_raw_shard(local_parquet, cache_dir):
    try:
        if local_parquet and os.path.exists(local_parquet):
            os.remove(local_parquet)
            print(f"Deleted raw Parquet shard {local_parquet}")
    finally:
        if cache_dir and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir, ignore_errors=True)


def get_worker_tokenizer(tokenizer_path):
    tokenizer = _TOKENIZER_CACHE.get(tokenizer_path)
    if tokenizer is None:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        _TOKENIZER_CACHE[tokenizer_path] = tokenizer
    return tokenizer


def tokenize_text_batch(batch, tokenizer_path, eos_id, context_length, text_column):
    tokenizer = get_worker_tokenizer(tokenizer_path)
    encodings = tokenizer.encode_batch_fast(batch[text_column])
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


def load_tokenized_blocks(tokenized_path):
    manifest_path = os.path.join(tokenized_path, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shards = []
    for shard in manifest["shards"]:
        shard_path = os.path.join(tokenized_path, shard["path"])
        shards.append(load_from_disk(shard_path))

    if not shards:
        raise ValueError(f"No tokenized shards found in {tokenized_path}")

    dataset = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    return dataset.with_format("torch"), manifest


def manifest_is_complete(tokenized_path, expected_label):
    manifest_path = os.path.join(tokenized_path, "manifest.json")
    if not os.path.exists(manifest_path):
        return False
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest.get("completed"):
        return False
    if manifest.get("label") != expected_label:
        return False
    for shard in manifest.get("shards", []):
        if not os.path.exists(os.path.join(tokenized_path, shard["path"])):
            return False
    return True


def build_or_load_tokenized_blocks(
    dataset_id,
    config_name,
    split,
    text_column,
    num_docs,
    tokenizer_path,
    tokenizer,
    vocab_size,
    special_tokens,
    context_length,
    tokenized_root,
    raw_parquet_dir,
    data_label=None,
    encode_workers=1,
    tokenize_batch_size=500,
):
    tokenizer_sha256 = file_sha256(tokenizer_path)
    label = data_label or default_data_label(
        dataset_id=dataset_id,
        config_name=config_name,
        split=split,
        text_column=text_column,
        num_docs=num_docs,
        vocab_size=vocab_size,
        context_length=context_length,
        tokenizer_sha256=tokenizer_sha256,
        special_tokens=special_tokens,
    )
    tokenized_path = os.path.join(tokenized_root, label)

    if manifest_is_complete(tokenized_path, label):
        print(f"Loading tokenized dataset '{label}' from {tokenized_path} ...")
        return load_tokenized_blocks(tokenized_path)

    eos_id = tokenizer.token_to_id(special_tokens[0])
    if eos_id is None:
        raise ValueError(f"Tokenizer is missing special token {special_tokens[0]!r}")

    print(f"Building tokenized dataset '{label}' in {tokenized_path} ...")
    os.makedirs(tokenized_path, exist_ok=True)
    shards_root = os.path.join(tokenized_path, "shards")
    os.makedirs(shards_root, exist_ok=True)
    shutil.copy2(tokenizer_path, os.path.join(tokenized_path, "tokenizer.json"))

    parquet_files = list_parquet_files(dataset_id, config_name)
    manifest = {
        "completed": False,
        "pipeline_version": TOKENIZED_PIPELINE_VERSION,
        "label": label,
        "dataset_id": dataset_id,
        "config_name": config_name,
        "split": split,
        "text_column": text_column,
        "num_docs_requested": num_docs,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "special_tokens": list(special_tokens),
        "tokenizer_sha256": tokenizer_sha256,
        "tokenizer_path": "tokenizer.json",
        "token_column": TOKEN_BLOCK_COLUMN,
        "shards": [],
        "docs_loaded": 0,
        "total_blocks": 0,
        "total_training_tokens": 0,
    }
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)

    docs_seen = 0
    total_blocks = 0
    for source_index, source_file in enumerate(parquet_files):
        if docs_seen >= num_docs:
            break

        local_parquet = download_raw_parquet(dataset_id, source_file, raw_parquet_dir)
        cache_dir = os.path.join(raw_parquet_dir, "_datasets_cache", f"tokenize_{source_index:05d}")
        ds = None
        ds_tok = None
        try:
            ds = load_dataset(
                "parquet",
                data_files=local_parquet,
                split=split,
                cache_dir=cache_dir,
            )
            if text_column not in ds.column_names:
                raise ValueError(
                    f"Expected column {text_column!r} in {source_file}, got {ds.column_names}"
                )

            source_docs = len(ds)
            remaining_docs = num_docs - docs_seen
            docs_in_shard = min(remaining_docs, source_docs)
            if docs_in_shard < source_docs:
                ds = ds.select(range(docs_in_shard))
            docs_seen += len(ds)

            num_proc = max(1, min(encode_workers, len(ds)))
            print(
                f"Tokenizing {len(ds):,} docs from {source_file} "
                f"with {num_proc} worker(s) ..."
            )
            ds_tok = ds.map(
                tokenize_text_batch,
                batched=True,
                batch_size=tokenize_batch_size,
                num_proc=num_proc,
                remove_columns=ds.column_names,
                fn_kwargs={
                    "tokenizer_path": tokenizer_path,
                    "eos_id": eos_id,
                    "context_length": context_length,
                    "text_column": text_column,
                },
                desc=f"Tokenizing {Path(source_file).name}",
            )

            shard_name = f"shard_{len(manifest['shards']):05d}"
            shard_path = os.path.join(shards_root, shard_name)
            if os.path.exists(shard_path):
                shutil.rmtree(shard_path)
            ds_tok.save_to_disk(shard_path)

            shard_blocks = len(ds_tok)
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
            del ds_tok
            del ds
            gc.collect()
            cleanup_raw_shard(local_parquet, cache_dir)

    if docs_seen < num_docs:
        print(f"Only found {docs_seen:,} docs before raw shards were exhausted")
    if total_blocks == 0:
        raise ValueError(
            "Tokenization produced zero training blocks. "
            "Use more documents or a shorter context length."
        )

    manifest["completed"] = True
    write_json_atomic(os.path.join(tokenized_path, "manifest.json"), manifest)
    with open(os.path.join(tokenized_path, "label.txt"), "w", encoding="utf-8") as f:
        f.write(label + "\n")
    print(f"Tokenized dataset label: {label}")
    return load_tokenized_blocks(tokenized_path)
