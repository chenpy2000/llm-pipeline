import torch
from torch.utils.data import DataLoader
import os
import csv
import json
import argparse
from datetime import datetime

from tokenizer import Tokenizer
from datasets import load_dataset
from dataset import LMDataset

from transformer import Decoder

# ── System ────────────────────────────────────────────────────────────────────
seed        = 42
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR       = "./data/fineweb-edu"
NUM_DOCS       = 500_000
VOCAB_SIZE     = 4000
SPECIAL_TOKENS = ["<|endoftext|>"]

# ── Model ─────────────────────────────────────────────────────────────────────
context_length = 256   # maximum sequence length
d_model        = 128   # embedding dimension
d_ff           = 512   # feedforward dimension (convention: 4 * d_model)
num_heads      = 4     # number of attention heads
num_layers     = 4     # number of transformer layers

# ── Training ──────────────────────────────────────────────────────────────────
batch_size     = 128
learning_rate  = 1e-3
eval_interval  = 50    # log every N steps
early_stop     = 0     # 0 to disable; stop after N evals with no val PPL improvement
token_budget   = 0     # 0 = disabled (epoch mode), >0 = Chinchilla mode

@torch.no_grad()
def compute_perplexity(decoderLMmodel, data_loader):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses = []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.1):
    """Autoregressive sampling from the decoder."""
    model.eval()
    token_ids = tokenizer.encode(prompt)
    x = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        # Crop to block_size if the sequence gets too long
        x_cond = x[:, -context_length:]
        logits, _ = model(x_cond)                     # no targets → returns logits
        logits = logits[:, -1, :] / temperature       # last position only
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    model.train()
    return tokenizer.decode(x.squeeze(0).tolist())

def load_data(data_dir=DATA_DIR, num_docs=NUM_DOCS):
    """
    Load FineWeb-EDU documents, downloading and caching locally on first run.

    Returns:
        list[str] — raw document texts (no separator tokens yet)
    """
    cache_path = os.path.join(data_dir, f"cached_{num_docs}")

    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path} ...")
        from datasets import load_from_disk
        ds = load_from_disk(cache_path)
    else:
        print(f"Downloading FineWeb-EDU ({num_docs:,} docs) ...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=f"train[:{num_docs}]",
            cache_dir=data_dir,
        )
        os.makedirs(cache_path, exist_ok=True)
        ds.save_to_disk(cache_path)
        print(f"Cached to {cache_path}")

    texts = ds["text"]
    print(f"Loaded {len(texts):,} documents")
    return texts

def encode_doc(args):
    text, tokenizer_path = args
    tok = Tokenizer.load(tokenizer_path)
    eos_id = tok.bytes_to_id[b"<|endoftext|>"]
    return tok.encode(text) + [eos_id]

def main():

    # ── Timestamp & output dir ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join("output", timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output → {run_dir}")

    print("Loading Dataset")
    raw_texts = load_data(data_dir=DATA_DIR, num_docs=NUM_DOCS)

    print("Loading Tokenizer")
    tokenizer_path = f"tokenizer/tokenizer_{VOCAB_SIZE}.json"
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path} (vocab size: {tokenizer.vocab_size})")
    else:
        print("No saved tokenizer found, training a new one ...")
        tokenizer = Tokenizer.train(raw_texts, vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
        os.makedirs("tokenizer", exist_ok=True)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path} (vocab size: {tokenizer.vocab_size})")

# ── Encode (cached) ──────────────────────────────────────────────────────
    encoded_dir = "encoded"
    os.makedirs(encoded_dir, exist_ok=True)
    encoded_path = os.path.join(encoded_dir, f"tokens_v{VOCAB_SIZE}_d{NUM_DOCS}.pt")

    if os.path.exists(encoded_path):
        print(f"Loading cached tokens from {encoded_path} ...")
        token_ids = torch.load(encoded_path)
    else:
        from multiprocessing import Pool, cpu_count
        n_workers = min(cpu_count(), 16)
        tokenizer_path = f"tokenizer/tokenizer_{VOCAB_SIZE}.json"
        work = [(text, tokenizer_path) for text in raw_texts]

        with Pool(n_workers) as pool:
            results = pool.map(encode_doc, work)

        token_ids = [tid for doc_ids in results for tid in doc_ids]
        torch.save(token_ids, encoded_path)
        print(f"Cached tokens to {encoded_path}")

    total_tokens = len(token_ids)
    print(f"Total tokens: {total_tokens:,}")

    # Train/val split (90/10)
    val_tokens = min(1_000_000, len(token_ids) // 10)  # ~1M tokens, or 10% for small datasets
    split = len(token_ids) - val_tokens
    train_dataset = LMDataset(token_ids[:split], context_length)
    val_dataset   = LMDataset(token_ids[split:], context_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    print(f"Train: {len(train_dataset):,} samples, Val: {len(val_dataset):,} samples")

    # Loading Model
    model = Decoder(vocab_size=tokenizer.vocab_size,
                    block_size=context_length,
                    d_model=d_model,
                    n_head=num_heads,
                    d_ff=d_ff,
                    n_layer=num_layers)
    
    print("Model Summary:")
    print(f"  Layers: {num_layers} | Heads: {num_heads} | Context: {context_length}")
    print(f"  d_model: {d_model} | d_ff: {d_ff}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Cosine LR scheduler (matched to token budget, or one epoch if no budget)
    total_steps_est = (token_budget // (batch_size * context_length)) if token_budget > 0 else len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_est, eta_min=learning_rate * 0.1)

    # ── Training log CSV ──────────────────────────────────────────────────────
    log_path = os.path.join(run_dir, f"training_log_{timestamp}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "total_steps", "loss", "train_ppl", "val_ppl", "lr"])

    print("Training ...")
    model.train()
    step = 0
    best_val_ppl = float("inf")
    no_improve = 0

    tokens_per_step = batch_size * context_length
    tokens_seen = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1
        tokens_seen += tokens_per_step
        if step % eval_interval == 0:
            train_ppl = torch.exp(torch.tensor(loss.item())).item()
            val_ppl   = compute_perplexity(model, val_loader)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Step {step}/{len(train_loader)} | "
                f"Tokens: {tokens_seen:,} | "
                f"LR: {current_lr:.2e} | "
                f"Loss: {loss.item():.4f} | "
                f"Train PPL: {train_ppl:.2f} | "
                f"Val PPL: {val_ppl:.2f}"
            )

            # Log to CSV
            log_writer.writerow([step, len(train_loader), f"{loss.item():.6f}",
                                 f"{train_ppl:.4f}", f"{val_ppl:.4f}", f"{current_lr:.6e}"])
            log_file.flush()

            # Early Stop configs below
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                no_improve = 0
            else:
                no_improve += 1

            if early_stop > 0 and no_improve >= early_stop:
                print(f"Early stopping at step {step} (no improvement for {early_stop} evals)")
                break

        if token_budget > 0 and tokens_seen >= token_budget:
            print(f"Token budget reached at step {step} ({tokens_seen:,} tokens)")
            break

    log_file.close()

    # Final eval
    train_ppl = compute_perplexity(model, train_loader)
    val_ppl   = compute_perplexity(model, val_loader)
    print(f"Final — Train PPL: {train_ppl:.2f} | Val PPL: {val_ppl:.2f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = os.path.join(run_dir, f"model_{timestamp}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_ppl": best_val_ppl,
    }, model_path)
    print(f"Model saved → {model_path}")

    # Generation
    prompts = [
        "The meaning of life is",
        "In the future, artificial intelligence will",
        "Education is important because",
    ]
    print("\n--- Generation ---")
    generation_outputs = []
    for prompt in prompts:
        output = generate(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")
        generation_outputs.append({"prompt": prompt, "output": output})

    # ── Save run config ───────────────────────────────────────────────────────
    run_config = {
        "timestamp": timestamp,
        "seed": seed,
        "device": str(device),

        "data": {
            "data_dir": DATA_DIR,
            "num_docs": NUM_DOCS,
            "total_tokens": total_tokens,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },

        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
        },

        "model": {
            "context_length": context_length,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "total_params": total_params,
        },

        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "eval_interval": eval_interval,
            "early_stop": early_stop,
            "token_budget": token_budget,
            "tokens_seen": tokens_seen,
            "final_step": step,
            "total_steps": len(train_loader),
            "best_val_ppl": best_val_ppl,
            "final_train_ppl": train_ppl,
            "final_val_ppl": val_ppl,
        },

        "generation": generation_outputs,
    }

    config_path = os.path.join(run_dir, f"run_config_{timestamp}.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config saved → {config_path}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model",      type=int, default=None)
    parser.add_argument("--num_layers",   type=int, default=None)
    parser.add_argument("--num_heads",    type=int, default=None)
    parser.add_argument("--d_ff",         type=int, default=None)
    parser.add_argument("--num_docs",     type=int, default=None)
    parser.add_argument("--early_stop",   type=int, default=None)
    parser.add_argument("--token_budget", type=int, default=None)
    args = parser.parse_args()

    # Override globals only if provided
    if args.d_model      is not None: d_model      = args.d_model
    if args.num_layers   is not None: num_layers   = args.num_layers
    if args.num_heads    is not None: num_heads    = args.num_heads
    if args.d_ff         is not None: d_ff         = args.d_ff
    if args.num_docs     is not None: NUM_DOCS     = args.num_docs
    if args.early_stop   is not None: early_stop   = args.early_stop
    if args.token_budget is not None: token_budget = args.token_budget

    main()
