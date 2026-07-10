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
N_SHOT = 1
PROMPT_TASK_IDS = (2, 3, 4)


def format_mbpp_task(example):
    """Use the task format from the MBPP evaluation protocol."""
    tests = "\n".join(example["test_list"])
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{example['text']} Your code should pass these tests:\n\n"
        f"{tests}\n[BEGIN]\n"
    )


def format_prompt(demonstrations, example):
    """Render MBPP few-shot examples inside the ChatML format used for SFT."""
    prompt = ""
    for demonstration in demonstrations:
        prompt += (
            "<|im_start|>user\n"
            f"{format_mbpp_task(demonstration)}"
            "<|im_end|>\n<|im_start|>assistant\n"
            f"{demonstration['code']}\n[DONE]<|im_end|>\n"
        )
    return (
        prompt
        + "<|im_start|>user\n"
        + format_mbpp_task(example)
        + "<|im_end|>\n<|im_start|>assistant\n"
    )


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
    prompt_dataset = load_dataset("google-research-datasets/mbpp", "full", split="prompt")
    prompt_examples = {example["task_id"]: example for example in prompt_dataset}
    demonstrations = [prompt_examples[task_id] for task_id in PROMPT_TASK_IDS[:N_SHOT]]
    passed = 0
    end_of_text_id = tokenizer.token_to_id("<|endoftext|>")

    with torch.inference_mode():
        for example in tqdm(dataset, desc="MBPP"):
            prompt = format_prompt(demonstrations, example)
            input_ids = tokenizer.encode(prompt).ids
            generated_ids = []
            for _ in range(MAX_NEW_TOKENS):
                logits = model(torch.tensor(input_ids[-CONTEXT_LENGTH:], device=device).unsqueeze(0))
                next_id = logits[0, -1].argmax().item()
                if next_id == end_of_text_id:
                    break
                input_ids.append(next_id)
                generated_ids.append(next_id)
                generated_text = tokenizer.decode(generated_ids)
                if generated_text.endswith("[DONE]") or generated_text.endswith("<|im_end|>"):
                    break

            code = tokenizer.decode(generated_ids).split("[DONE]", 1)[0].split("<|im_end|>", 1)[0]
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
