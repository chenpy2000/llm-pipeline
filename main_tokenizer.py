import argparse
import gc
import json
import os
from datetime import datetime

from datasets import load_dataset

from data_pipeline import (
    cleanup_raw_shard,
    download_raw_parquet,
    file_sha256,
    list_parquet_files,
    write_json_atomic,
)
from mix_dataset import (
    cleanup_raw_source_shard,
    download_raw_source_file,
    extract_text,
    list_source_files,
    load_raw_source_dataset,
    normalize_cpt_source,
    source_file_format,
    source_to_manifest,
    validate_source_columns,
)
from tokenizer import Tokenizer
from tokenizer_config import DEFAULT_VOCAB_SIZE, SPECIAL_TOKENS, default_tokenizer_path


FINEWEB_DATASET_ID = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_SPLIT = "train"
FINEWEB_TEXT_COLUMN = "text"
FINEWEB_DOC_CAP = 1_000_000

RAW_DATA_DIR = "./data/raw_tokenizer"
ITERATOR_BATCH_SIZE = 1000

DEFAULT_CPT_DOC_CAP = 100_000
DEFAULT_CPT_DOC_CAPS = {
    "python_edu_cleaned": 200_000,
    "codeparrot_clean": 100_000,
    "starcoder_python": 100_000,
    "starcoder_jupyter_scripts": 50_000,
    "starcoder_jupyter_structured": 50_000,
    "starcoder_github_issues": 50_000,
    "starcoder_git_commits": 50_000,
    "opc_fineweb_code": 200_000,
    "finemath_4plus": 100_000,
}
EXCLUDED_CPT_LABELS = {"fineweb_edu_replay"}


def count_local_text_lines(text_files):
    total = 0
    for path in text_files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def load_sources_json(path):
    with open(path, "r", encoding="utf-8") as f:
        sources = json.load(f)
    if not isinstance(sources, list):
        raise ValueError("Tokenizer sources JSON must be a list of CPT source objects")
    return sources


def default_cpt_sources():
    from main_cpt import CPT_SOURCES

    return [
        source
        for source in CPT_SOURCES
        if source.get("label") not in EXCLUDED_CPT_LABELS
    ]


def cpt_doc_cap(source, max_docs_per_cpt_source):
    if max_docs_per_cpt_source is not None:
        return max_docs_per_cpt_source
    return DEFAULT_CPT_DOC_CAPS.get(source.label, DEFAULT_CPT_DOC_CAP)


def iter_fineweb_texts(
    raw_data_dir,
    max_docs,
    batch_size,
    stats,
):
    if max_docs <= 0:
        return

    parquet_files = list_parquet_files(FINEWEB_DATASET_ID, FINEWEB_CONFIG)
    docs_seen = 0
    for shard_index, source_file in enumerate(parquet_files):
        if docs_seen >= max_docs:
            break

        local_parquet = download_raw_parquet(
            FINEWEB_DATASET_ID,
            source_file,
            raw_data_dir,
        )
        cache_dir = os.path.join(
            raw_data_dir,
            "_datasets_cache",
            f"tokenizer_fineweb_{shard_index:05d}",
        )
        ds = None
        try:
            ds = load_dataset(
                "parquet",
                data_files=local_parquet,
                split=FINEWEB_SPLIT,
                cache_dir=cache_dir,
            )
            if FINEWEB_TEXT_COLUMN not in ds.column_names:
                raise ValueError(
                    f"Expected column {FINEWEB_TEXT_COLUMN!r} in {source_file}, "
                    f"got {ds.column_names}"
                )

            remaining_docs = max_docs - docs_seen
            docs_in_shard = min(remaining_docs, len(ds))
            if docs_in_shard < len(ds):
                ds = ds.select(range(docs_in_shard))

            stats["files_visited"].append(source_file)
            stats["docs_consumed"] += len(ds)
            docs_seen += len(ds)

            for batch in ds.iter(batch_size=batch_size):
                for text in batch[FINEWEB_TEXT_COLUMN]:
                    if text:
                        stats["texts_yielded"] += 1
                        yield text
        finally:
            ds = None
            gc.collect()
            cleanup_raw_shard(local_parquet, cache_dir)


def iter_cpt_source_texts(
    source,
    raw_data_dir,
    hf_token,
    max_docs,
    batch_size,
    stats,
):
    if max_docs <= 0:
        return

    source_files = list_source_files(source, hf_token=hf_token)
    stats["source_files"] = source_files
    docs_seen = 0

    for source_index, source_file in enumerate(source_files):
        if docs_seen >= max_docs:
            break

        local_file = None
        ds = None
        cache_dir = os.path.join(
            raw_data_dir,
            "_datasets_cache",
            f"tokenizer_{source.label}_{source_index:05d}",
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

            remaining_docs = max_docs - docs_seen
            docs_in_shard = min(remaining_docs, len(ds))
            if docs_in_shard < len(ds):
                ds = ds.select(range(docs_in_shard))

            stats["files_visited"].append(source_file)
            stats["docs_consumed"] += len(ds)
            docs_seen += len(ds)

            for batch in ds.iter(batch_size=batch_size):
                row_count = len(next(iter(batch.values()))) if batch else 0
                for row_index in range(row_count):
                    example = {
                        column: values[row_index]
                        for column, values in batch.items()
                    }
                    text = extract_text(example, source)
                    if text:
                        stats["texts_yielded"] += 1
                        yield text
        finally:
            ds = None
            gc.collect()
            cleanup_raw_source_shard(local_file, cache_dir)


def iter_tokenizer_texts(args, manifest):
    if args.text_file:
        local_stats = {
            "dataset_id": "local_text_files",
            "files_visited": [],
            "docs_consumed": 0,
            "texts_yielded": 0,
        }
        manifest["sources"].append(local_stats)
        for path in args.text_file:
            local_stats["files_visited"].append(path)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.rstrip("\r\n")
                    if text.strip():
                        local_stats["docs_consumed"] += 1
                        local_stats["texts_yielded"] += 1
                        yield text

    fineweb_stats = {
        "dataset_id": FINEWEB_DATASET_ID,
        "config_name": FINEWEB_CONFIG,
        "split": FINEWEB_SPLIT,
        "text_column": FINEWEB_TEXT_COLUMN,
        "doc_cap": args.fineweb_docs,
        "files_visited": [],
        "docs_consumed": 0,
        "texts_yielded": 0,
    }
    manifest["sources"].append(fineweb_stats)
    yield from iter_fineweb_texts(
        raw_data_dir=args.raw_data_dir,
        max_docs=args.fineweb_docs,
        batch_size=args.iterator_batch_size,
        stats=fineweb_stats,
    )

    raw_sources = load_sources_json(args.sources_json) if args.sources_json else default_cpt_sources()
    for raw_source in raw_sources:
        source = normalize_cpt_source(raw_source)
        if source.label in EXCLUDED_CPT_LABELS:
            continue

        max_docs = cpt_doc_cap(source, args.max_docs_per_cpt_source)
        source_stats = {
            **source_to_manifest(source),
            "doc_cap": max_docs,
            "source_files": [],
            "files_visited": [],
            "docs_consumed": 0,
            "texts_yielded": 0,
        }
        manifest["sources"].append(source_stats)
        yield from iter_cpt_source_texts(
            source=source,
            raw_data_dir=args.raw_data_dir,
            hf_token=args.hf_token,
            max_docs=max_docs,
            batch_size=args.iterator_batch_size,
            stats=source_stats,
        )


def estimate_iterator_length(args, sources):
    total = count_local_text_lines(args.text_file)
    total += max(0, args.fineweb_docs)
    for raw_source in sources:
        source = normalize_cpt_source(raw_source)
        if source.label in EXCLUDED_CPT_LABELS:
            continue
        total += max(0, cpt_doc_cap(source, args.max_docs_per_cpt_source))
    return total


def train_tokenizer(args):
    output_path = args.output_path or default_tokenizer_path(args.vocab_size)
    manifest_path = args.manifest_path or f"{os.path.splitext(output_path)[0]}_manifest.json"

    if os.path.exists(output_path) and not args.overwrite:
        raise FileExistsError(
            f"Tokenizer already exists at {output_path}. Pass --overwrite to replace it."
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(args.raw_data_dir, exist_ok=True)

    raw_sources = load_sources_json(args.sources_json) if args.sources_json else default_cpt_sources()
    iterator_length = estimate_iterator_length(args, raw_sources)
    if iterator_length <= 0:
        raise ValueError("Tokenizer training needs at least one input text")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": output_path,
        "vocab_size_requested": args.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "raw_data_dir": args.raw_data_dir,
        "iterator_batch_size": args.iterator_batch_size,
        "fineweb_docs": args.fineweb_docs,
        "max_docs_per_cpt_source": args.max_docs_per_cpt_source,
        "estimated_iterator_length": iterator_length,
        "sources": [],
    }

    tokenizer = Tokenizer()
    trainer = Tokenizer.build_bpe_trainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(
        iter_tokenizer_texts(args, manifest),
        trainer=trainer,
        length=iterator_length,
    )
    tokenizer.save(output_path)

    manifest["vocab_size_actual"] = tokenizer.vocab_size
    manifest["tokenizer_sha256"] = file_sha256(output_path)
    manifest["total_docs_consumed"] = sum(
        int(source["docs_consumed"]) for source in manifest["sources"]
    )
    manifest["total_texts_yielded"] = sum(
        int(source["texts_yielded"]) for source in manifest["sources"]
    )
    write_json_atomic(manifest_path, manifest)

    print(f"Tokenizer saved -> {output_path} (vocab size: {tokenizer.vocab_size:,})")
    print(f"Tokenizer manifest saved -> {manifest_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--raw_data_dir", type=str, default=RAW_DATA_DIR)
    parser.add_argument("--fineweb_docs", type=int, default=FINEWEB_DOC_CAP)
    parser.add_argument("--max_docs_per_cpt_source", type=int, default=None)
    parser.add_argument("--sources_json", type=str, default=None)
    parser.add_argument("--text_file", action="append", default=[])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--iterator_batch_size", type=int, default=ITERATOR_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_tokenizer(parse_args())
