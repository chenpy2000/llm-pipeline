"""Run MBPP pass@1 evaluation on a saved checkpoint.

Usage:
    uv run python benchmark_coding.py
"""

import subprocess
import sys
import tempfile

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from tokenizer import Tokenizer
from transformer import Decoder


TOKENIZER_PATH = "tokenizer/tokenizer_32768.json"
MODEL_PATH = "output_1024/model_sft_coding/model_sft_coding.pt"
VOCAB_SIZE = 32768
CONTEXT_LENGTH = 1024
D_MODEL = 896
SWIGLU_D = 4864
NUM_HEADS = 14
NUM_KEY_VALUE_HEADS = 2
NUM_LAYERS = 24
ROPE_BASE = 1_000_000.0
MAX_NEW_TOKENS = 256


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = Decoder(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=NUM_HEADS,
        n_kv_head=NUM_KEY_VALUE_HEADS,
        swiglu_d=SWIGLU_D,
        n_layer=NUM_LAYERS,
        rope_base=ROPE_BASE,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = load_dataset("google-research-datasets/mbpp", "full", split="test")
    passed = 0
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")

    with torch.inference_mode():
        for example in tqdm(dataset, desc="MBPP"):
            prompt = (
                "<|im_start|>user\n"
                "Write a Python solution for this task. Return only executable Python code.\n\n"
                f"{example['text']}"
                "<|im_end|>\n<|im_start|>assistant\n"
            )
            input_ids = tokenizer.encode(prompt).ids
            generated_ids = []
            for _ in range(MAX_NEW_TOKENS):
                logits = model(torch.tensor(input_ids[-CONTEXT_LENGTH:], device=device).unsqueeze(0))
                next_id = logits[0, -1].argmax().item()
                if next_id == end_of_text_id:
                    break
                input_ids.append(next_id)
                generated_ids.append(next_id)
                if tokenizer.decode(generated_ids).endswith("<|im_end|>"):
                    break

            code = tokenizer.decode(generated_ids).split("<|im_end|>", 1)[0]
            if "```" in code:
                code = code.split("```", 2)[1].removeprefix("python").lstrip()
            script = "\n".join([example["test_setup_code"], code, *example["test_list"]])
            try:
                with tempfile.TemporaryDirectory() as directory:
                    result = subprocess.run(
                        [sys.executable, "-I", "-c", script],
                        cwd=directory,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                    )
                passed += result.returncode == 0
            except subprocess.TimeoutExpired:
                pass

    total = len(dataset)
    print(f"MBPP pass@1: {passed / total:.2%} ({passed}/{total})")


if __name__ == "__main__":
    main()
