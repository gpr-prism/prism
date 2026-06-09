#!/usr/bin/env python3
"""
GDN — Gated Delta Net

Base model: GLA decay gate + DeltaNet directional erasure, no solver.
This is the baseline that EFLA, PGDN, and PRISM all build upon.

Update rule:
  S_t = gk_t * (S_{t-1} - beta_t * k_t k_t^T S_{t-1}) + k_t v_t^T
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ModelConfig


class GDNBlock(nn.Module):
    """
    GDN (Gated Delta Net) block for language modelling.

    Structure: single-layer UVQK projection + GLA decay + DeltaNet erasure.
    No solver components.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        D = cfg.embed_dim
        H = cfg.num_heads
        self._num_heads = H
        self._linear_dim = D // H
        self._attention_dim = D // H
        self._max_seq_len = cfg.seq_len
        self._chunk_size = min(128, cfg.seq_len)
        self._eps = 1e-3
        self._gate_low_rank_dim = cfg.gate_low_rank_dim
        self._gate_temperature = cfg.gate_temperature
        self._gate_min = cfg.gate_min

        self._dropout = nn.Dropout(p=cfg.dropout)

        # Single-layer Q/K/V + U projection
        self._uvqk = nn.Linear(D, self._linear_dim * 2 * H + self._attention_dim * H * 2)

        # Short conv for local context (anchor)
        conv_size = cfg.short_kernel_size
        self._conv_size = conv_size
        self.short_conv = nn.Conv1d(D, D, kernel_size=conv_size, groups=D, bias=True)
        self.anchor_norm = nn.LayerNorm(D)

        # Output projection
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * H, D * 4), nn.SiLU(), nn.Linear(D * 4, D),
        )

        # GLA per-element decay gate (low-rank)
        self._gate = nn.Sequential(
            nn.Linear(D, self._gate_low_rank_dim),
            nn.Linear(self._gate_low_rank_dim, self._attention_dim * H),
            nn.Sigmoid(),
        )

        self.layer_norm_input = nn.LayerNorm(D)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * H)
        self._g = nn.Linear(D, self._linear_dim * H)

        # DeltaNet beta: directional erasure strength
        self.b_proj = nn.Linear(D, H, bias=False)

    def _short_conv_anchor(self, x):
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self._conv_size - 1, 0))
        anchor = self.short_conv(x_pad)[..., :x_t.size(-1)]
        return anchor.transpose(1, 2)

    def _recurrence(self, q, k_base, v_base, gk, beta, initial_state):
        """GLA chunk-parallel recurrence + DeltaNet directional erasure."""
        B = q.shape[0]
        num_chunks = self._max_seq_len // self._chunk_size
        C = self._chunk_size
        H = self._num_heads
        d_k = self._attention_dim
        d_v = self._linear_dim
        invalid_attn_mask = torch.tril(torch.ones(C, C, device=q.device))

        q_c = q.reshape(B, num_chunks, C, H, d_k)
        k_c = k_base.reshape(B, num_chunks, C, H, d_k)
        v_c = v_base.reshape(B, num_chunks, C, H, d_v)
        gk_c = gk.reshape(B, num_chunks, C, H, d_k)
        beta_c = beta.reshape(B, num_chunks, C, H)

        orig_dtype = gk_c.dtype
        gk_c = gk_c.float()
        q_c = q_c.float()
        k_c = k_c.float()
        v_c = v_c.float()
        beta_c = beta_c.float()
        initial_state = initial_state.float()

        log_gk_c = torch.log(gk_c.clamp(min=1e-6))
        log_decay_start = torch.cumsum(log_gk_c, dim=2).clamp(min=-20.0, max=20.0)
        decay_start = torch.exp(log_decay_start)
        log_decay_end = torch.flip(
            torch.cumsum(torch.flip(log_gk_c, dims=[2]), dim=2), dims=[2]
        ).clamp(min=-20.0, max=20.0)
        decay_end = torch.exp(log_decay_end)
        log_prod_gk = torch.sum(log_gk_c, dim=2).clamp(min=-20.0, max=20.0)
        prod_gk = torch.exp(log_prod_gk)
        gkv = torch.einsum('bnha,l->bnhal', prod_gk, torch.ones(d_v, device=gk_c.device))

        # Base KV accumulation
        k_de = k_c * decay_end
        base_kv = torch.einsum('bnsha,bnshl->bnhal', k_de, v_c)

        # DeltaNet directional erasure (chunk-level mean key)
        k_mean = k_c.mean(dim=2)
        k_mean = k_mean / (k_mean.norm(dim=-1, keepdim=True) + 1e-6)
        beta_mean = beta_c.mean(dim=2)

        _MEM_MAX = 50.0
        memory_states = [initial_state]
        mem = initial_state
        for i in range(num_chunks - 1):
            km = k_mean[:, i]
            bm = beta_mean[:, i].unsqueeze(-1).unsqueeze(-1)
            kmTS = torch.einsum('bha,bhav->bhv', km, mem).unsqueeze(2)
            km_kmTS = km.unsqueeze(-1) * kmTS
            mem = mem - bm * km_kmTS
            mem = mem * gkv[:, i]
            mem = mem + base_kv[:, i]
            mem = mem.clamp(min=-_MEM_MAX, max=_MEM_MAX)
            memory_states.append(mem)

        memory_states = torch.reshape(
            torch.cat(memory_states, dim=1), [B, num_chunks, H, d_k, d_v]
        )

        q_ds = q_c * decay_start
        o_inter = torch.einsum('bnsha,bnhal->bnshl', q_ds, memory_states)

        k_ds = k_c / (decay_start + self._eps).clamp(min=1e-6)
        p = torch.einsum('bnshl,bnmhl->bnhsm', q_ds, k_ds).clamp(min=-100.0, max=100.0)
        p_masked = p * invalid_attn_mask[:C, :C]
        o_intra = torch.einsum('bnhsm,bnmhl->bnshl', p_masked, v_c)

        o = o_inter + o_intra
        return o.reshape(B, self._max_seq_len, H, d_v).to(orig_dtype)

    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            return self._forward_impl(x)

    def _forward_impl(self, x):
        normed_x = self.layer_norm_input(x)
        B, T = normed_x.shape[0], normed_x.shape[1]

        anchor = self._short_conv_anchor(normed_x)
        anchor = self.anchor_norm(anchor)

        out = self._uvqk(anchor)
        u, v_base, q, k_base = torch.split(out, [
            self._linear_dim * self._num_heads,
            self._linear_dim * self._num_heads,
            self._attention_dim * self._num_heads,
            self._attention_dim * self._num_heads,
        ], dim=-1)

        gk = self._gate(normed_x)
        beta = self.b_proj(normed_x).sigmoid()

        if self._gate_temperature != 1.0:
            gk = torch.exp(torch.log(gk.clamp(min=1e-6)) / self._gate_temperature)
        gk = gk.clamp(min=max(self._gate_min, 0.01), max=1.0)

        q_h = q.view(B, self._max_seq_len, self._num_heads, self._attention_dim)
        k_h = k_base.view(B, self._max_seq_len, self._num_heads, self._attention_dim)
        v_h = v_base.view(B, self._max_seq_len, self._num_heads, self._linear_dim)

        initial_state = torch.zeros(
            B, self._num_heads, self._attention_dim, self._linear_dim, device=x.device
        )

        attn_output = self._recurrence(q_h, k_h, v_h, gk, beta, initial_state)

        g = F.silu(self._g(normed_x).reshape(
            -1, self._max_seq_len, self._num_heads, self._linear_dim
        ))
        attn_output = g * attn_output
        attn_output = attn_output.reshape(
            -1, self._max_seq_len, self._num_heads * self._linear_dim
        )
        o_input = u * self.layer_norm_output(attn_output)

        x = self._o(self._dropout(o_input)) + x
        return x
