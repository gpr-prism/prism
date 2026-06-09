#!/usr/bin/env python3
"""
Shared components for PRISM and baseline models.
"""

import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Unified configuration for all models."""
    vocab_size: int = 50257
    embed_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    seq_len: int = 256
    dropout: float = 0.1
    _config_name: str = "small"
    # PRISM-specific
    solver_steps: int = 2
    short_kernel_size: int = 5
    gate_low_rank_dim: int = 16
    gate_temperature: float = 16.0
    gate_min: float = 1e-4
    solver_out_init: float = -2.0
    # DeltaNet-specific (kept for compatibility)
    deltanet_expand_v: float = 1.0
    deltanet_conv_size: int = 4


CONFIGS = {
    "large": ModelConfig(
        embed_dim=768, num_layers=8, num_heads=8, seq_len=1024, dropout=0.1,
        solver_steps=2, short_kernel_size=7, _config_name="large",
    ),
    "large_130m": ModelConfig(
        embed_dim=768, num_layers=8, num_heads=8, seq_len=1024, dropout=0.1,
        solver_steps=2, short_kernel_size=7, _config_name="large_130m",
    ),
    "xlarge": ModelConfig(
        embed_dim=2048, num_layers=14, num_heads=16, seq_len=2048, dropout=0.1,
        solver_steps=2, short_kernel_size=7, _config_name="xlarge",
    ),
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """Low-rank factorised linear: D_in -> rank -> D_out."""
    def __init__(self, d_in, d_out, rank=None, bias=False):
        super().__init__()
        if rank is None:
            rank = max(16, d_in // 4)
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up = nn.Linear(rank, d_out, bias=bias)

    def forward(self, x):
        return self.up(self.down(x))


# ---------------------------------------------------------------------------
# CausalLM wrapper (shared by all block types)
# ---------------------------------------------------------------------------

class CausalLM(nn.Module):
    """Causal language model wrapper around any block class."""

    def __init__(self, cfg: ModelConfig, block_cls):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
