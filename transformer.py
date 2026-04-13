import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, masked=False):
        super(MultiHeadAttention, self).__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.masked = masked
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.W_q = nn.Linear(n_embd, n_embd)
        self.W_k = nn.Linear(n_embd, n_embd)
        self.W_v = nn.Linear(n_embd, n_embd)
        self.W_o = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # x: B x T x n_embd
        B, T, _ = x.shape

        Q = self.W_q(x)  # B x T x n_embd
        K = self.W_k(x)  # B x T x n_embd
        V = self.W_v(x)  # B x T x n_embd

        Q = Q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim
        K = K.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim
        V = V.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # B x n_head x T x head_dim    

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
    def __init__(self, d_model, n_head, d_ff, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, masked=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN 
        attn_output, attn_probs = self.attn(self.ln1(x))  # B x T x n_embd, B x n_head x T x T
        x = x + attn_output
        x = x + self.ff(self.ln2(x))  # B x T x n_out

        # Post-LN
        # attn_output, attn_probs = self.attn(x)
        # x = x + attn_output
        # x = self.ln1(x)
        # ff_output = self.ff(x)
        # x = x + ff_output
        # x = self.ln2(x)

        return x, attn_probs

class Decoder(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_head, d_ff, n_layer):
        super(Decoder, self).__init__()
        self.n_embd = d_model
        self.tok_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pos_emb = nn.Embedding(num_embeddings=block_size, embedding_dim=d_model)
        self.blocks = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_head=n_head, d_ff=d_ff)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # x: B x T
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        out = self.tok_emb(x) + self.pos_emb(pos)

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