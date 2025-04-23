import torch
from torch import nn
from torch import Tensor

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class GovnArgs:
    dim: int = 512
    n_layers: int = 16
    n_heads: int = 8
    vocab_size: int = -1
    feed_forward_hidden_dim: int = 2048
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 256

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.y = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.y * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    

class Swish(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(self.beta * x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.swish = Swish()

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.swish(self.w1(x)) * self.w3(x))

def get_precomputed_rotary_angles(dim: int, seq_len: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    d_half = dim // 2
    theta = torch.pow(base, -2 * torch.arange(0, d_half, dtype=torch.float32) / dim)
    pos = torch.arange(0, seq_len, dtype=torch.float32)
    theta = pos.view(-1, 1) @ theta.view(1, -1)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin
    
def apply_rotary_embedding(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    res = torch.zeros_like(x)

    res[..., ::2] += x[..., ::2] * cos
    res[..., 1::2] += x[..., 1::2] * cos

    res[..., ::2] += -x[..., 1::2] * sin
    res[..., 1::2] += x[..., ::2] * sin

    return res


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wout = nn.Linear(dim, dim)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q_rotated = apply_rotary_embedding(q, cos, sin)
        k_rotated = apply_rotary_embedding(k, cos, sin)

        # (batch_size, seq_len, n_heads, head_dim)
        q_rotated = q_rotated.view(q_rotated.shape[0], q_rotated.shape[1], self.n_heads, self.head_dim)
        k_rotated = k_rotated.view(k_rotated.shape[0], k_rotated.shape[1], self.n_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.n_heads, self.head_dim)
        
        attn = torch.matmul(q_rotated.permute(0, 2, 1, 3), k_rotated.permute(0, 2, 3, 1)) / torch.sqrt(torch.tensor(self.head_dim))
        assert attn.shape == (x.shape[0], self.n_heads, x.shape[1], x.shape[1])

        if mask is not None:
            attn += mask

        attn = nn.functional.softmax(attn, dim=-1)

        # (batch_size, n_heads, seq_len, head_dim)
        out = torch.matmul(attn, v.permute(0, 2, 1, 3))
        # (batch_size, seq_len, n_heads, head_dim)
        out = out.permute(0, 2, 1, 3)

        out = torch.flatten(out, start_dim=2)

        return self.wout(out)
    

class DecoderBlock(nn.Module):
    def __init__(self, args: GovnArgs):
        super().__init__()
        self.args = args
        self.norm1 = RMSNorm(args.dim)
        self.attn = Attention(args.dim, args.n_heads)
        self.norm2 = RMSNorm(args.dim)
        self.ff = FeedForward(args.dim, args.feed_forward_hidden_dim)  # Исправлено

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, cos, sin, mask)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        return x
    

class GovnLM(nn.Module):
    def __init__(self, args: GovnArgs):
        super().__init__()

        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([DecoderBlock(args) for _ in range(args.n_layers)])
        self.out_proj = nn.Linear(args.dim, args.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[1] <= self.args.max_seq_len
        seq_len = x.shape[1]
        cos, sin = get_precomputed_rotary_angles(self.args.dim, seq_len)
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(x.device)

        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.out_proj(x)