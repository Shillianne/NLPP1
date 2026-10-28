import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import TypeVar, Generic
from src.moe.latent_moe import LatentMoe


@dataclass
class Config:
    vocab_size: int = 50257
    context_length: int = 1024
    batch_size: int = 2
    drop_rate: float = 0.0
    qkv_bias: bool = True
    emb_dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_experts: int = 8
    top_k: int = 2
    bias: bool = False
    d: int = emb_dim
    m: int = 256
    latent: int = 256
    n_shared_exp: int = 1


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape

        keys: torch.Tensor = self.W_key(x)
        queries: torch.Tensor = self.W_query(x)
        values: torch.Tensor = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)  # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore
        attn_scores.masked_fill_(mask_bool, -torch.inf)  # modifies attn_scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class Base:
    ...


Kind = TypeVar("Kind", Base, LatentMoe)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config, ff: Kind = Base):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )
        if ff == Base:
            self.ff = FeedForward(cfg)
        else:
            self.ff = LatentMoe(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_resid = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        
        return x


class GPTModel(Generic[Kind], nn.Module):
    def __init__(self, cfg: Config, ff: Kind = Base):
        super(GPTModel, self).__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg, ff=ff) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params, latent: bool = False):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.pos_emb.weight.requires_grad = False
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])
    gpt.tok_emb.weight.requires_grad = False

    for b in range(len(gpt.trf_blocks)):
        q_w, k_w, v_w = torch.chunk(
            params[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        gpt.trf_blocks[b].att.W_query.weight.requires_grad = False
        gpt.trf_blocks[b].att.W_key.weight.requires_grad = False
        gpt.trf_blocks[b].att.W_value.weight.requires_grad = False

        q_b, k_b, v_b = torch.chunk(
            params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.W_query.bias.requires_grad = False
        gpt.trf_blocks[b].att.W_key.bias.requires_grad = False
        gpt.trf_blocks[b].att.W_value.bias.requires_grad = False

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"h.{b}.attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"h.{b}.attn.c_proj.bias"])
        gpt.trf_blocks[b].att.out_proj.weight.requires_grad = False
        gpt.trf_blocks[b].att.out_proj.bias.requires_grad = False

        if not latent:
            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params[f"h.{b}.mlp.c_fc.weight"].T)
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params[f"h.{b}.mlp.c_fc.bias"])
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params[f"h.{b}.mlp.c_proj.weight"].T)
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params[f"h.{b}.mlp.c_proj.bias"])
            gpt.trf_blocks[b].ff.layers[0].weight.requires_grad = False
            gpt.trf_blocks[b].ff.layers[0].bias.requires_grad = False
            gpt.trf_blocks[b].ff.layers[2].weight.requires_grad = False
            gpt.trf_blocks[b].ff.layers[2].bias.requires_grad = False

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params[f"h.{b}.ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params[f"h.{b}.ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params[f"h.{b}.ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params[f"h.{b}.ln_2.bias"])
        gpt.trf_blocks[b].norm1.scale.requires_grad = False
        gpt.trf_blocks[b].norm1.shift.requires_grad = False
        gpt.trf_blocks[b].norm2.scale.requires_grad = False
        gpt.trf_blocks[b].norm2.shift.requires_grad = False

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])
    gpt.final_norm.scale.requires_grad = False
    gpt.final_norm.shift.requires_grad = False
    gpt.out_head.weight.requires_grad = False