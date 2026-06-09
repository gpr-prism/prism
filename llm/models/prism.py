#!/usr/bin/env python3
"""
PRISM - Parallel Residual Iterative Sequence Model

GDN + lightweight low-rank solver (<10% parameter overhead).

Architecture:
  1. Base path (identical to GDN):
       u, v, q, k = Linear(anchor)
       gk = gate(normed_x),  beta = sigmoid(b_proj(normed_x))
  2. Solver path (low-rank, from anchor):
       For each step l:
         P^(l) = LowRank_P^(l)(anchor)   — gain modulator
         K^(l) = LowRank_K^(l)(anchor)   — solver key
         beta^(l) = sigma(Linear(anchor)) — per-step gate
         delta^(l) = P^(l) * retain * v  — closed-form value refinement
         retain *= (1 - P^(l))
  3. Recurrence: S = erasure(S) -> gk*S -> + base_kv + solver_B_extra
     where B_extra = sum_l beta^(l) * (K^(l) outer delta^(l))

Key design choices:
  - NO solver_v_proj: residual = v_base (shared with main path)
  - NO solver_out: solver deltas injected directly into state update
  - ALL solver projections are LOW-RANK (rank=128 by default)
  - Closed-form decomposition: delta^(l) = P^(l) * prod_{j<l}(1-P^(j)) * v
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ModelConfig


class PRISMBlock(nn.Module):
    """
    PRISM block for language modelling.
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
        self._solver_steps = cfg.solver_steps

        self._dropout = nn.Dropout(p=cfg.dropout)

        # Solver low-rank config
        self._lr_rank = 128
        self._num_solver_proj = self._solver_steps * 2   # gain + k_proj per step
        self._gate_out_dim = H * self._solver_steps
        self._solver_down_dim = self._lr_rank * self._num_solver_proj + self._gate_out_dim

        # Fused projection: base KV + solver down-projections in one GEMM
        # Layout: [U | V | Q | K | gain_down×S | k_down×S | gate×S]
        self._base_proj_dim = self._linear_dim * 2 * H + self._attention_dim * H * 2
        self._fused_proj = nn.Linear(D, self._base_proj_dim + self._solver_down_dim, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self._fused_proj.bias[self._base_proj_dim:])
            self._fused_proj.bias.data[
                self._base_proj_dim + self._lr_rank * self._num_solver_proj:
            ] = 2.0

        # Short conv anchor
        conv_size = cfg.short_kernel_size
        self._conv_size = conv_size
        self.short_conv = nn.Conv1d(D, D, kernel_size=conv_size, groups=D, bias=True)
        self.anchor_norm = nn.LayerNorm(D)

        # Output projection
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * H, D * 4), nn.SiLU(), nn.Linear(D * 4, D),
        )

        # GLA per-element decay gate
        self._gate = nn.Sequential(
            nn.Linear(D, self._gate_low_rank_dim),
            nn.Linear(self._gate_low_rank_dim, self._attention_dim * H),
            nn.Sigmoid(),
        )

        self.layer_norm_input = nn.LayerNorm(D)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * H)
        self._g = nn.Linear(D, self._linear_dim * H)

        # DeltaNet beta
        self.b_proj = nn.Linear(D, H, bias=False)

        # Solver up-projections (batched matmul via parameter tensors)
        self.solver_up_gain_weight = nn.Parameter(
            torch.empty(self._solver_steps, self._lr_rank, D)
        )
        self.solver_up_k_weight = nn.Parameter(
            torch.empty(self._solver_steps, self._lr_rank, self._attention_dim * H)
        )
        for i in range(self._solver_steps):
            nn.init.kaiming_uniform_(self.solver_up_gain_weight[i], a=5**0.5)
            nn.init.kaiming_uniform_(self.solver_up_k_weight[i], a=5**0.5)

    def _short_conv_anchor(self, x):
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (self._conv_size - 1, 0))
        anchor = self.short_conv(x_pad)[..., :x_t.size(-1)]
        return anchor.transpose(1, 2)

    def _split_fused_proj(self, fused_out):
        base_out = fused_out[..., :self._base_proj_dim]
        solver_down_out = fused_out[..., self._base_proj_dim:]
        u, v_base, q, k_base = torch.split(base_out, [
            self._linear_dim * self._num_heads,
            self._linear_dim * self._num_heads,
            self._attention_dim * self._num_heads,
            self._attention_dim * self._num_heads,
        ], dim=-1)
        return u, v_base, q, k_base, solver_down_out

    def _solver_projections_from_down(self, solver_down_out):
        """Compute solver projections from the solver portion of fused output."""
        R = self._lr_rank
        S = self._solver_steps
        gain_downs = solver_down_out[..., :R * S].chunk(S, dim=-1)
        k_downs = solver_down_out[..., R * S:R * S * 2].chunk(S, dim=-1)
        gate_outs = solver_down_out[..., R * S * 2:].chunk(S, dim=-1)

        gain_input = torch.stack(gain_downs, dim=0).reshape(S, -1, R)
        gain_out = torch.bmm(gain_input, self.solver_up_gain_weight)
        gains = [torch.sigmoid(gain_out[i].reshape_as(
            solver_down_out[..., :self.solver_up_gain_weight.shape[2]]
        )) for i in range(S)]

        k_input = torch.stack(k_downs, dim=0).reshape(S, -1, R)
        k_out = torch.bmm(k_input, self.solver_up_k_weight)
        keys = [k_out[i].reshape_as(
            solver_down_out[..., :self.solver_up_k_weight.shape[2]]
        ) for i in range(S)]

        gates = [torch.sigmoid(gate_outs[i]) for i in range(S)]
        return gains, keys, gates

    def _compute_solver_steps_from_down(self, solver_down_out, v_base):
        """
        Closed-form solver:
          delta^(l) = P_l * retain * v_base
          retain *= (1 - P_l)
        """
        B, T = v_base.shape[0], v_base.shape[1]
        R = self._lr_rank
        S = self._solver_steps
        H = self._num_heads

        gain_downs = solver_down_out[..., :R * S].chunk(S, dim=-1)
        k_downs = solver_down_out[..., R * S:R * S * 2].chunk(S, dim=-1)
        gate_outs = solver_down_out[..., R * S * 2:].chunk(S, dim=-1)

        D = self.solver_up_gain_weight.shape[2]
        gain_input = torch.stack(gain_downs, dim=0).reshape(S, B * T, R)
        gain_out = torch.bmm(gain_input, self.solver_up_gain_weight)
        P_list = [torch.sigmoid(gain_out[i].view(B, T, D)) for i in range(S)]

        d_kH = self.solver_up_k_weight.shape[2]
        k_input = torch.stack(k_downs, dim=0).reshape(S, B * T, R)
        k_out = torch.bmm(k_input, self.solver_up_k_weight)

        all_k_list, all_delta_list, all_beta_list = [], [], []
        retain = torch.ones_like(v_base)

        for step in range(S):
            P_l = P_list[step]
            beta_l = torch.sigmoid(gate_outs[step])
            k_l = k_out[step].view(B, T, H, self._attention_dim)

            delta = P_l * retain * v_base
            retain = retain * (1 - P_l)

            all_k_list.append(k_l)
            all_delta_list.append(delta)
            all_beta_list.append(beta_l)

        return all_k_list, all_delta_list, all_beta_list

    def _recurrence(self, q, k_base, v_base, gk, beta,
                    initial_state, all_k_list, all_delta_list, all_beta_list):
        """
        GLA chunk-parallel recurrence + DeltaNet erasure + solver injection.
        Solver deltas are added directly into the state update (no separate output path).
        """
        B = q.shape[0]
        num_chunks = self._max_seq_len // self._chunk_size
        C = self._chunk_size
        H = self._num_heads
        d_k = self._attention_dim
        d_v = self._linear_dim
        S = self._solver_steps

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
        log_prod_gk = log_gk_c.sum(dim=2).clamp(min=-20.0, max=20.0)
        prod_gk = torch.exp(log_prod_gk)
        gkv = prod_gk.unsqueeze(-1).expand(-1, -1, -1, -1, d_v)

        k_de = k_c * decay_end
        base_kv = torch.einsum('bnsha,bnshl->bnhal', k_de, v_c)

        # Vectorized solver B_extra computation
        solver_k_chunks, solver_v_chunks = [], []
        if S > 0:
            all_k_stacked = torch.stack([kl.float() for kl in all_k_list], dim=0)
            all_delta_stacked = torch.stack([
                dl.float().view(B, self._max_seq_len, H, d_v) for dl in all_delta_list
            ], dim=0)
            all_beta_stacked = torch.stack([bl.float() for bl in all_beta_list], dim=0)

            all_k_c = all_k_stacked.reshape(S, B, num_chunks, C, H, d_k)
            all_delta_c = all_delta_stacked.reshape(S, B, num_chunks, C, H, d_v)
            all_beta_c = all_beta_stacked.reshape(S, B, num_chunks, C, H)

            all_k_de = all_k_c * decay_end.unsqueeze(0)
            all_delta_w = all_delta_c * all_beta_c.unsqueeze(-1)

            Sk = all_k_de.reshape(S * B, num_chunks, C, H, d_k)
            Sv = all_delta_w.reshape(S * B, num_chunks, C, H, d_v)
            B_extra_all = torch.einsum('bnsha,bnshl->bnhal', Sk, Sv)
            B_extra = B_extra_all.reshape(S, B, num_chunks, H, d_k, d_v).sum(dim=0)

            for l in range(S):
                solver_k_chunks.append(all_k_c[l])
                solver_v_chunks.append(all_delta_w[l])
        else:
            B_extra = torch.zeros(B, num_chunks, H, d_k, d_v, device=q.device, dtype=torch.float32)

        chunk_update = base_kv + B_extra

        k_mean = k_c.mean(dim=2)
        k_mean = k_mean / (k_mean.norm(dim=-1, keepdim=True) + 1e-6)
        beta_mean = beta_c.mean(dim=2)

        _MEM_MAX = 50.0
        memory_states = [initial_state]
        mem = initial_state
        for i in range(num_chunks - 1):
            km = k_mean[:, i]
            bm = beta_mean[:, i, :, None, None]
            proj = (km.unsqueeze(-1) * mem).sum(dim=-2, keepdim=True)
            mem = mem - bm * km.unsqueeze(-1) * proj
            mem = mem * gkv[:, i] + chunk_update[:, i]
            mem = mem.clamp(min=-_MEM_MAX, max=_MEM_MAX)
            memory_states.append(mem)

        memory_states = torch.stack(memory_states, dim=1)

        q_ds = q_c * decay_start
        o_inter = torch.einsum('bnsha,bnhal->bnshl', q_ds, memory_states)

        # Intra-chunk: combine base K/V with solver K/V (element-wise addition)
        invalid_attn_mask = torch.tril(torch.ones(C, C, device=q.device))
        k_combined = k_c
        v_combined = v_c
        if S > 0:
            for l in range(S):
                k_combined = k_combined + solver_k_chunks[l]
                v_combined = v_combined + solver_v_chunks[l]

        inv_decay = 1.0 / (decay_start + self._eps).clamp(min=1e-6)
        k_ds = k_combined * inv_decay
        p = torch.einsum('bnshl,bnmhl->bnhsm', q_ds, k_ds).clamp(min=-100.0, max=100.0)
        p_masked = p * invalid_attn_mask[:C, :C]
        o_intra = torch.einsum('bnhsm,bnmhl->bnshl', p_masked, v_combined)

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

        fused_out = self._fused_proj(anchor)
        u, v_base, q, k_base, solver_down_out = self._split_fused_proj(fused_out)

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

        all_k_list, all_delta_list, all_beta_list = self._compute_solver_steps_from_down(
            solver_down_out, v_base
        )

        attn_output = self._recurrence(
            q_h, k_h, v_h, gk, beta, initial_state,
            all_k_list, all_delta_list, all_beta_list
        )

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
