from __future__ import annotations

from typing import Any


CONFIG: dict[str, dict[str, Any]] = {
    "tokenizer": {
        "vocab_size": 32768,
        "special_tokens": [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            # "<|fim_pad|>",
            # "<|repo_name|>",
            # "<|file_sep|>",
        ],
    },
    "model": {
        "architecture_reference": "Qwen2.5-Coder-0.5B",
        "context_length": 32768,
        "d_model": 896,
        "swiglu_d": 4864,
        "num_heads": 14,
        "num_key_value_heads": 2,
        "num_layers": 24,
        "rope_base": 1000000.0,
    },
    "training": {
        "batch_size": 4,
        "learning_rate": 0.0003,
        "eval_interval": 1000,
        "dtype": "bfloat16",
        "val_tokens": 262144,
    },
}


def load_config() -> dict[str, dict[str, Any]]:
    return CONFIG


def get_config_section(section: str) -> dict[str, Any]:
    config = load_config()
    try:
        value = config[section]
    except KeyError as exc:
        raise KeyError(f"Missing config section {section!r} in config.py") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Config section {section!r} must be an object")
    return value


def torch_dtype(dtype_name: str):
    import torch

    name = str(dtype_name).replace("torch.", "")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype in config: {dtype_name!r}")
    return dtype
