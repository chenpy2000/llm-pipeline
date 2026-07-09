"""Run zero-shot ARC-Challenge evaluation on a saved checkpoint.

Usage:
    uv run python benchmark_reasoning.py
"""

import torch
import torch.nn.functional as F
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

    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    correct = 0

    with torch.inference_mode():
        for example in tqdm(dataset, desc="ARC-Challenge"):
            choices = example["choices"]
            prompt = (
                "<|im_start|>user\n"
                f"{example['question']}\n\n"
                "Choose the correct answer. Reply with only its letter.\n"
                + "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
                + "<|im_end|>\n<|im_start|>assistant\nAnswer:"
            )
            prompt_ids = tokenizer.encode(prompt).ids
            answer_ids = [tokenizer.encode(f" {label}").ids for label in choices["label"]]
            input_ids = [
                prompt_ids[-(CONTEXT_LENGTH - len(answer)):] + answer[:-1]
                for answer in answer_ids
            ]
            logits = model(torch.tensor(input_ids, device=device))
            targets = torch.tensor(answer_ids, device=device)
            log_probs = F.log_softmax(logits[:, -targets.size(1):], dim=-1)
            scores = log_probs.gather(2, targets.unsqueeze(2)).sum(dim=1)
            prediction = choices["label"][scores.argmax().item()]
            correct += prediction == example["answerKey"]

    total = len(dataset)
    print(f"ARC-Challenge 0-shot test accuracy: {correct / total:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
