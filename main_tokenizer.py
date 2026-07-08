import argparse
import json
import os
from datetime import datetime

from mix_dataset import (
    file_sha256,
    iter_mixed_source_texts,
    normalize_hf_source,
    write_json_atomic,
)
from config import get_config_section
from tokenizer import Tokenizer


TOKENIZER_CONFIG = get_config_section("tokenizer")
DEFAULT_VOCAB_SIZE = int(TOKENIZER_CONFIG["vocab_size"])
SPECIAL_TOKENS = list(TOKENIZER_CONFIG["special_tokens"])

RAW_DATA_DIR = "./data/raw_tokenizer"
ITERATOR_BATCH_SIZE = 1000

TOKENIZER_SOURCES = [
    {
        "label": "fineweb_edu",
        "dataset_id": "HuggingFaceFW/fineweb-edu",
        "config_name": "sample-10BT",
        "split": "train",
        "text_column": "text",
        "file_format": "parquet",
        "token_budget": 7_000_000_000,
        "tokenizer_docs": 1_000_000,
    },
    {
        "label": "python_edu_cleaned",
        "dataset_id": "Avelina/python-edu-cleaned",
        "config_name": "default",
        "split": "train",
        "text_column": "text",
        "token_budget": 2_500_000_000,
        "fim_rate": 0.30,
        "tokenizer_docs": 200_000,
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
        "tokenizer_docs": 100_000,
    },
    {
        "label": "starcoder_python",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "python",
        "split": "train",
        "text_column": "content",
        "token_budget": 250_000_000,
        "fim_rate": 0.30,
        "tokenizer_docs": 100_000,
    },
    {
        "label": "starcoder_jupyter_scripts",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "jupyter-scripts-dedup-filtered",
        "split": "train",
        "text_column": "content",
        "token_budget": 100_000_000,
        "fim_rate": 0.30,
        "tokenizer_docs": 50_000,
    },
    {
        "label": "starcoder_jupyter_structured",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "jupyter-structured-clean-dedup",
        "split": "train",
        "text_columns": ["content", "text", "code", "markdown"],
        "token_budget": 50_000_000,
        "fim_rate": 0.30,
        "tokenizer_docs": 50_000,
    },
    {
        "label": "starcoder_github_issues",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "github-issues-filtered-structured",
        "split": "train",
        "text_columns": ["title", "body", "comments", "text", "content"],
        "token_budget": 50_000_000,
        "fim_rate": 0.0,
        "tokenizer_docs": 50_000,
    },
    {
        "label": "starcoder_git_commits",
        "dataset_id": "bigcode/starcoderdata",
        "data_dir": "git-commits-cleaned",
        "split": "train",
        "text_columns": ["message", "content", "diff", "text"],
        "token_budget": 50_000_000,
        "fim_rate": 0.15,
        "tokenizer_docs": 50_000,
    },
    {
        "label": "opc_fineweb_code",
        "dataset_id": "OpenCoder-LLM/opc-fineweb-code-corpus",
        "config_name": "default",
        "split": "train",
        "text_column": "text",
        "token_budget": 400_000_000,
        "fim_rate": 0.30,
        "tokenizer_docs": 200_000,
    },
    {
        "label": "finemath_4plus",
        "dataset_id": "HuggingFaceTB/finemath",
        "config_name": "finemath-4plus",
        "split": "train",
        "text_column": "text",
        "token_budget": 300_000_000,
        "fim_rate": 0.0,
        "tokenizer_docs": 100_000,
    },
]


def default_tokenizer_path(vocab_size=DEFAULT_VOCAB_SIZE):
    return os.path.join("tokenizer", f"tokenizer_{vocab_size}.json")


def validate_tokenizer(
    tokenizer,
    tokenizer_path,
    expected_vocab_size=DEFAULT_VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    strict_vocab_size=True,
):
    if strict_vocab_size and tokenizer.vocab_size != expected_vocab_size:
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has vocab size {tokenizer.vocab_size:,}, "
            f"but this run expects {expected_vocab_size:,}. Run "
            "`uv run python main_tokenizer.py` to create the configured tokenizer."
        )

    missing_tokens = [
        token for token in special_tokens if tokenizer.token_to_id(token) is None
    ]
    if missing_tokens:
        raise ValueError(
            f"Tokenizer at {tokenizer_path} is missing required special tokens: "
            f"{missing_tokens}. Run `uv run python main_tokenizer.py` first."
        )


def load_required_tokenizer(
    tokenizer_path,
    expected_vocab_size=DEFAULT_VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
):
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Run "
            "`uv run python main_tokenizer.py` first."
        )

    tokenizer = Tokenizer.from_file(tokenizer_path)
    validate_tokenizer(
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
        expected_vocab_size=expected_vocab_size,
        special_tokens=special_tokens,
        strict_vocab_size=True,
    )
    return tokenizer


def validate_checkpoint_vocab(checkpoint, expected_vocab_size, checkpoint_path):
    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain model_state_dict."
        )

    embedding = state_dict.get("tok_emb.weight")
    if embedding is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain tok_emb.weight."
        )

    checkpoint_vocab_size = int(embedding.shape[0])
    if checkpoint_vocab_size != expected_vocab_size:
        raise ValueError(
            f"Checkpoint vocab mismatch for {checkpoint_path}: checkpoint has "
            f"{checkpoint_vocab_size:,} embedding rows, but the active tokenizer "
            f"expects {expected_vocab_size:,}."
        )


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
        raise ValueError("Tokenizer sources JSON must be a list of HF source objects")
    return sources


def tokenizer_hf_sources(args):
    return load_sources_json(args.sources_json) if args.sources_json else TOKENIZER_SOURCES


def tokenizer_doc_caps(sources, args):
    caps = {}
    for raw_source in sources:
        source = normalize_hf_source(raw_source)
        cap = args.max_docs_per_source
        if cap is None and isinstance(raw_source, dict):
            cap = raw_source.get("tokenizer_docs")
        if cap is None:
            cap = source.num_docs
        caps[source.label] = cap
    return caps


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

    sources = tokenizer_hf_sources(args)
    yield from iter_mixed_source_texts(
        sources=sources,
        raw_data_dir=args.raw_data_dir,
        hf_token=args.hf_token,
        max_docs_by_label=tokenizer_doc_caps(sources, args),
        batch_size=args.iterator_batch_size,
        stats_list=manifest["sources"],
        apply_fim=False,
        cache_namespace="tokenizer",
    )


def estimate_iterator_length(args, sources):
    total = count_local_text_lines(args.text_file)
    for raw_source in sources:
        source = normalize_hf_source(raw_source)
        cap = tokenizer_doc_caps([source], args).get(source.label)
        if cap is not None:
            total += max(0, cap)
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

    sources = tokenizer_hf_sources(args)
    iterator_length = estimate_iterator_length(args, sources)
    if iterator_length <= 0:
        raise ValueError("Tokenizer training needs at least one input text")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_path": output_path,
        "vocab_size_requested": args.vocab_size,
        "special_tokens": list(SPECIAL_TOKENS),
        "raw_data_dir": args.raw_data_dir,
        "iterator_batch_size": args.iterator_batch_size,
        "max_docs_per_source": args.max_docs_per_source,
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
    parser.add_argument("--max_docs_per_source", type=int, default=None)
    parser.add_argument("--sources_json", type=str, default=None)
    parser.add_argument("--text_file", action="append", default=[])
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--iterator_batch_size", type=int, default=ITERATOR_BATCH_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train_tokenizer(parse_args())
