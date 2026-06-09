#!/usr/bin/env python3
"""
PGDN — Preconditioned Gated Delta Net

GDN + ATK (Adaptive Trace-norm Key) preconditioning:
  A_t = alpha_atk * A_{t-1} + beta_atk * k_t^2   (diagonal accumulator)
  k_precond = k * sqrt(A_t)                        (preconditioned key)

Key insight: k for READING (erasure direction), k_precond for WRITING (state update).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ModelConfig


class PGDNBlock(nn.Module):
    """
    PGDN (Preconditioned Gated Delta Net) block.

    Recurrence:
      ATK:  A_t = alpha_atk * A_{t-1} + beta_atk * k^2
            k_precond = k * sqrt(A_t)
      Delta Rule:
            S_t = gk * (S_{t-1} - beta * k k^T S_{t-1}) + k_precond v^T
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
        self._eps = 1e-6
        self._gate_low_rank_dim = cfg.gate_low_rank_dim
        self._gate_temperature = cfg.gate_temperature
        self._gate_min = cfg.gate_min

        self._dropout = nn.Dropout(p=cfg.dropout)

        self._uvqk = nn.Linear(D, self._linear_dim * 2 * H + self._attention_dim * H * 2)

        conv_size = cfg.short_kernel_size
        self._conv_size = conv_size
        self.short_conv = nn.Conv1d(D, D, kernel_size=conv_size, groups=D, bias=True)
        self.anchor_norm = nn.LayerNorm(D)

        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * H, D * 4), nn.SiLU(), nn.Linear(D * 4, D),
        )

        self._gate = nn.Sequential(
            nn.Linear(D, self._gate_low_rank_dim),
            nn.Linear(self._gate_low_rank_dim, self._attention_dim * H),
            nn.Sigmoid(),
        )

        self.layer_norm_input = nn.LayerNorm(D)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * H)
        self._g = nn.Linear(D, self._linear_dim * H)

        self.b_proj = nn.Linear(D, H, bias=False)

        # ATK-specific projections
        self._g_atk_proj = nn.Linear(D, H)
        self._beta_atk_proj = nn.Linear(D, H)
        self._log_A_scale = nn.Parameter(torch.full((H,), -1.0))

    def _short_conv_anchor(self, x):
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self._conv_size - 1, 0))
        anchor = self.short_conv(x_pad)[..., :x_t.size(-1)]
        return anchor.transpose(1, 2)

    def _fused_atk_fwd(self, k_flat, beta_atk, g_atk):
        """
        ATK forward: A_t = alpha_atk * A_{t-1} + beta_atk * k^2
        Returns k_precond = k * sqrt(A_t + eps).
        Uses parallel scan within chunks for efficiency.
        """
        B, T, H, d_k = k_flat.shape
        C = self._chunk_size
        num_chunks = T // C

        alpha_atk = torch.sigmoid(g_atk)
        k_sq = k_flat ** 2

        alpha_c = alpha_atk.reshape(B, num_chunks, C, H)
        beta_atk_expanded = beta_atk.unsqueeze(-1).expand_as(k_sq)
        bk2 = (beta_atk_expanded * k_sq).reshape(B, num_chunks, C, H, d_k)

        log_alpha = torch.log(alpha_c.clamp(min=1e-6))
        cum_log_alpha = torch.cumsum(log_alpha, dim=2)
        decay_from_start = torch.exp(cum_log_alpha.clamp(min=-20.0, max=20.0))

        inv_decay = torch.exp(-cum_log_alpha.clamp(min=-20.0, max=20.0))
        scaled_inputs = bk2 * inv_decay.unsqueeze(-1)
        cum_scaled = torch.cumsum(scaled_inputs, dim=2)

        initial_atk = torch.exp(self._log_A_scale).unsqueeze(0).unsqueeze(-1).expand(B, H, d_k)
        ac = torch.zeros(B, num_chunks + 1, H, d_k, device=k_flat.device, dtype=k_flat.dtype)
        ac[:, 0] = initial_atk
        for chunk_idx in range(num_chunks):
            chunk_decay_total = decay_from_start[:, chunk_idx, -1]
            chunk_cum_last = cum_scaled[:, chunk_idx, -1]
            ac[:, chunk_idx + 1] = chunk_decay_total.unsqueeze(-1) * (
                ac[:, chunk_idx] + chunk_cum_last
            )

        ac_start = ac[:, :num_chunks]
        a_atk = decay_from_start.unsqueeze(-1) * (ac_start.unsqueeze(2) + cum_scaled)
        a_atk = a_atk.reshape(B, T, H, d_k)

        k_precond = k_flat * torch.sqrt(a_atk + self._eps)
        return k_precond, ac[:, 1:], a_atk

    def _recurrence(self, q, k, k_precond, v, gk, beta, initial_state):
        """
        PGDN chunk-parallel recurrence.
        k for reading (erasure), k_precond for writing (state update).
        """
        B = q.shape[0]
        num_chunks = self._max_seq_len // self._chunk_size
        C = self._chunk_size
        H = self._num_heads
        d_k = self._attention_dim
        d_v = self._linear_dim
        invalid_attn_mask = torch.tril(torch.ones(C, C, device=q.device))

        q_c = q.reshape(B, num_chunks, C, H, d_k)
        k_c = k.reshape(B, num_chunks, C, H, d_k)
        kp_c = k_precond.reshape(B, num_chunks, C, H, d_k)
        v_c = v.reshape(B, num_chunks, C, H, d_v)
        gk_c = gk.reshape(B, num_chunks, C, H, d_k)
        beta_c = beta.reshape(B, num_chunks, C, H)

        orig_dtype = gk_c.dtype
        gk_c = gk_c.float()
        q_c = q_c.float()
        k_c = k_c.float()
        kp_c = kp_c.float()
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

        # Write with k_precond
        kp_de = kp_c * decay_end
        base_kv = torch.einsum('bnsha,bnshl->bnhal', kp_de, v_c)

        # Erase with k (reading key)
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

        # Intra-chunk uses k_precond for local attention
        kp_ds = kp_c / (decay_start + self._eps).clamp(min=1e-6)
        p = torch.einsum('bnshl,bnmhl->bnhsm', q_ds, kp_ds).clamp(min=-100.0, max=100.0)
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
        g_atk = self._g_atk_proj(normed_x)
        beta_atk = torch.sigmoid(self._beta_atk_proj(normed_x))

        if self._gate_temperature != 1.0:
            gk = torch.exp(torch.log(gk.clamp(min=1e-6)) / self._gate_temperature)
        gk = gk.clamp(min=max(self._gate_min, 0.01), max=1.0)

        q_h = q.view(B, self._max_seq_len, self._num_heads, self._attention_dim)
        k_h = k_base.view(B, self._max_seq_len, self._num_heads, self._attention_dim)
        v_h = v_base.view(B, self._max_seq_len, self._num_heads, self._linear_dim)

        initial_state = torch.zeros(
            B, self._num_heads, self._attention_dim, self._linear_dim, device=x.device
        )

        k_precond, _, _ = self._fused_atk_fwd(k_h, beta_atk, g_atk)

        attn_output = self._recurrence(q_h, k_h, k_precond, v_h, gk, beta, initial_state)

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
