import os

from tokenizer import Tokenizer


DEFAULT_VOCAB_SIZE = 151_936
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
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
            "`uv run python main_tokenizer.py` to create the 151,936-vocab tokenizer."
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
            f"expects {expected_vocab_size:,}. Old 32K-vocab checkpoints cannot "
            "be reused after the 151,936-vocab restart."
        )
