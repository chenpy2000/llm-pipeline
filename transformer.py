import torch
import torch.nn as nn


def apply_rope(x, cos, sin):
    # x: B x n_head x T x head_dim
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_even * cos - x_odd * sin
    x_rotated[..., 1::2] = x_even * sin + x_odd * cos
    return x_rotated


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, masked=False, rope_base=10000.0):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.masked = masked
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim % 2 == 0, "RoPE requires even head dimension"
        self.W_q = nn.Linear(n_embd, n_embd)
        self.W_k = nn.Linear(n_embd, n_embd)
        self.W_v = nn.Linear(n_embd, n_embd)
        self.W_o = nn.Linear(n_embd, n_embd)
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        # x: B x T x n_embd
        B, T, _ = x.shape

        Q = self.W_q(x)  # B x T x n_embd
        K = self.W_k(x)  # B x T x n_embd
        V = self.W_v(x)  # B x T x n_embd

        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim    

        positions = torch.arange(T, device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(positions, self.rope_inv_freq)
        cos = freqs.cos()[None, None, :, :].to(dtype=Q.dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=Q.dtype)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # B x n_head x T x T

        if self.masked:
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # B x n_head x T x T
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)  # B x n_head x T x T
        attn_output = torch.matmul(attn_probs, V)  # B x n_head x T x head_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.n_embd)  # B x T x n_embd
        attn_output = self.W_o(attn_output)

        return attn_output, attn_probs

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.0, rope_base=10000.0):
        super(DecoderLayer, self).__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, masked=True, rope_base=rope_base)
        self.ln2 = nn.RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, attn_probs = self.attn(self.ln1(x))  # B x T x n_embd, B x n_head x T x T
        x = x + attn_output
        x = x + self.ff(self.ln2(x))  # B x T x n_out

        return x, attn_probs

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, n_layer, rope_base=10000.0):
        super(Decoder, self).__init__()
        self.n_embd = d_model
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff, rope_base=rope_base)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.tok_emb.weight
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x: B x T
        out = self.tok_emb(x)

        att_maps = []
        for block in self.blocks:
            out, probs = block(out)  # B x T x n_embd, B x n_head x T x T
            att_maps.append(probs.mean(dim=1))  # Average attention probabilities over heads, resulting in T x T

        out = self.ln_f(out)
        logits = self.lm_head(out)  # B x T x vocab_size
        
        # Inference Only
        if y is None:
            return logits, att_maps

        # Loss Computation
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
        return loss
