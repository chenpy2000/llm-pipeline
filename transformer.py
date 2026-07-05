import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x, cos, sin):
    # x: B x n_head x T x head_dim
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = x_even * cos - x_odd * sin
    x_rotated[..., 1::2] = x_even * sin + x_odd * cos
    return x_rotated


def repeat_kv(x, n_rep):
    B, n_kv_head, T, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(B, n_kv_head, n_rep, T, head_dim)
    return x.reshape(B, n_kv_head * n_rep, T, head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head=None, masked=False, rope_base=10000.0):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        if n_kv_head is None:
            n_kv_head = n_head
        assert n_head % n_kv_head == 0, "Number of attention heads must be divisible by KV heads"
        self.masked = masked
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_rep = n_head // n_kv_head
        self.head_dim = n_embd // n_head
        assert self.head_dim % 2 == 0, "RoPE requires even head dimension"
        self.W_q = nn.Linear(n_embd, n_head * self.head_dim)
        self.W_k = nn.Linear(n_embd, n_kv_head * self.head_dim)
        self.W_v = nn.Linear(n_embd, n_kv_head * self.head_dim)
        self.W_o = nn.Linear(n_embd, n_embd)
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def forward(self, x):
        # x: B x T x n_embd
        B, T, _ = x.shape

        Q = self.W_q(x)  # B x T x (n_head * head_dim)
        K = self.W_k(x)  # B x T x (n_kv_head * head_dim)
        V = self.W_v(x)  # B x T x (n_kv_head * head_dim)

        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim
        K = K.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # B x n_kv_head x T x head_dim
        V = V.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # B x n_kv_head x T x head_dim

        positions = torch.arange(T, device=x.device, dtype=self.rope_inv_freq.dtype)
        freqs = torch.outer(positions, self.rope_inv_freq)
        cos = freqs.cos()[None, None, :, :].to(dtype=Q.dtype)
        sin = freqs.sin()[None, None, :, :].to(dtype=Q.dtype)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=0.0,
            is_causal=self.masked,
        )  # B x n_head x T x head_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.n_embd)  # B x T x n_embd
        attn_output = self.W_o(attn_output)

        return attn_output


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, swiglu_d, dropout=0.0):
        super(SwiGLUFFN, self).__init__()
        self.up_gate = nn.Linear(d_model, 2 * swiglu_d)
        self.act = nn.SiLU()
        self.down = nn.Linear(swiglu_d, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        values, gates = self.up_gate(x).chunk(2, dim=-1)
        x = values * self.act(gates)
        x = self.down(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, swiglu_d, dropout=0.0, rope_base=10000.0):
        super(DecoderLayer, self).__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, n_kv_head=n_kv_head, masked=True, rope_base=rope_base)
        self.ln2 = nn.RMSNorm(d_model)
        self.ff = SwiGLUFFN(d_model, swiglu_d, dropout=dropout)

    def forward(self, x):
        attn_output = self.attn(self.ln1(x))  # B x T x n_embd
        x = x + attn_output
        x = x + self.ff(self.ln2(x))  # B x T x n_out

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_kv_head, swiglu_d, n_layer, rope_base=10000.0):
        super(Decoder, self).__init__()
        self.n_embd = d_model
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_head=n_head, n_kv_head=n_kv_head, swiglu_d=swiglu_d, rope_base=rope_base)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.tok_emb.weight
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x: B x T
        out = self.tok_emb(x)

        for block in self.blocks:
            out = block(out)  # B x T x n_embd

        out = self.ln_f(out)
        logits = self.lm_head(out)  # B x T x vocab_size
        
        # Inference Only
        if y is None:
            return logits

        # Loss Computation
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(B * T, V), y.view(B * T))
        return loss
