"""A small decoder-only transformer, written to be inspected rather than hidden.

Everything PostHog's AI research listing names is visible here in plain PyTorch:
multi-head causal self-attention, pre-layer-norm blocks, GELU feed-forward layers
with a SwiGLU variant, weight tying, and a forward pass that can return the
attention probabilities so the behaviour can be analysed rather than assumed.

    from tinygpt import TinyGPT, ByteTokenizer
    model = TinyGPT(vocab=256, d_model=128, heads=4, layers=4)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTokenizer:
    """Byte-level tokenisation: 256 symbols, no vocabulary to hide behind."""

    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(int(i) for i in ids).decode("utf-8", "replace")


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0, swiglu=False):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must divide by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.swiglu = swiglu
        self.last_attention = None

    def forward(self, x, return_attention=False):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.heads, self.head_dim)

        def split(t):
            return t.view(*shape).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))
        attention = self.dropout(F.softmax(scores, dim=-1))
        if return_attention:
            self.last_attention = attention.detach()
        out = (attention @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.proj(out)


class FeedForward(nn.Module):
    """GELU MLP, or a SwiGLU variant when swiglu=True."""

    def __init__(self, d_model, hidden=None, dropout=0.0, swiglu=False):
        super().__init__()
        hidden = hidden or 4 * d_model
        self.swiglu = swiglu
        if swiglu:
            self.gate = nn.Linear(d_model, hidden, bias=False)
            self.up = nn.Linear(d_model, hidden, bias=False)
            self.down = nn.Linear(hidden, d_model, bias=False)
        else:
            self.up = nn.Linear(d_model, hidden)
            self.down = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.swiglu:
            return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0, swiglu=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads, dropout, swiglu=swiglu)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout, swiglu=swiglu)

    def forward(self, x, return_attention=False):
        x = x + self.attn(self.ln1(x), return_attention=return_attention)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab=256, d_model=128, heads=4, layers=4, context=96,
                 dropout=0.0, swiglu=False, tie_weights=True):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(Block(d_model, heads, dropout, swiglu=swiglu)
                                    for _ in range(layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False, return_attention=False):
        batch, length = idx.shape
        if length > self.context:
            raise ValueError("sequence longer than the model context")
        positions = torch.arange(length, device=idx.device)
        x = self.tok(idx) + self.pos(positions)
        for block in self.blocks:
            x = block(x, return_attention=return_attention)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        if return_hidden:
            return logits, loss, x
        return logits, loss

    def attention_maps(self, idx):
        """Run a forward pass and return every layer's attention probabilities."""
        maps = []
        self.eval()
        with torch.no_grad():
            self.forward(idx, return_attention=True)
            for block in self.blocks:
                maps.append(block.attn.last_attention)
        return maps

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
