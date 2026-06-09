#!/usr/bin/env python3
"""
PRISM Ablation Variants

All variants inherit from PRISMBlock and override only the solver step computation.

r2: Shared K projection across all solver steps
r3: No retain (no closed-form cumulative product)
r4: Step 0 reuses K_base from main path
r5: ALL steps reuse K_base (no independent K projections)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ModelConfig
from .prism import PRISMBlock


class PRISMr2Block(PRISMBlock):
    """
    PRISM Ablation r2 — SHARED K projection across all solver steps.

    All steps share step 0's K projection.
    Tests whether per-step independent keys are necessary.
    """

    def _compute_solver_steps_from_down(self, solver_down_out, v_base):
        B, T = v_base.shape[0], v_base.shape[1]
        gains, keys, gates = self._solver_projections_from_down(solver_down_out)

        all_k_list, all_delta_list, all_beta_list = [], [], []
        k_shared = keys[0].view(B, T, self._num_heads, self._attention_dim)
        retain = torch.ones_like(v_base)

        for step in range(self._solver_steps):
            P_l = gains[step]
            beta_l = gates[step]
            delta = P_l * retain * v_base
            retain = retain * (1 - P_l)
            all_k_list.append(k_shared)   # Same K for all steps
            all_delta_list.append(delta)
            all_beta_list.append(beta_l)

        return all_k_list, all_delta_list, all_beta_list


class PRISMr3Block(PRISMBlock):
    """
    PRISM Ablation r3 — NO retain (no closed-form cumulative product).

    Each step uses P_l * v_base directly without the (1-P_j) product.
    Tests whether the closed-form decomposition is important.
    """

    def _compute_solver_steps_from_down(self, solver_down_out, v_base):
        B, T = v_base.shape[0], v_base.shape[1]
        gains, keys, gates = self._solver_projections_from_down(solver_down_out)

        all_k_list, all_delta_list, all_beta_list = [], [], []

        for step in range(self._solver_steps):
            P_l = gains[step]
            beta_l = gates[step]
            k_l = keys[step].view(B, T, self._num_heads, self._attention_dim)
            delta = P_l * v_base   # No retain
            all_k_list.append(k_l)
            all_delta_list.append(delta)
            all_beta_list.append(beta_l)

        return all_k_list, all_delta_list, all_beta_list


class PRISMr4Block(PRISMBlock):
    """
    PRISM Ablation r4 — Step 0 reuses K_base from main path.

    Step 0 uses K_base instead of its own step_k_proj[0].
    Steps 1+ still use independent per-step K projections.
    Tests whether the solver's first rank update benefits from sharing K_base.
    """

    def _compute_solver_steps_with_k_base(self, solver_down_out, v_base, k_base_h):
        B, T = v_base.shape[0], v_base.shape[1]
        gains, keys, gates = self._solver_projections_from_down(solver_down_out)

        all_k_list, all_delta_list, all_beta_list = [], [], []
        retain = torch.ones_like(v_base)

        for step in range(self._solver_steps):
            P_l = gains[step]
            beta_l = gates[step]
            if step == 0:
                k_l = k_base_h   # Reuse K_base for step 0
            else:
                k_l = keys[step].view(B, T, self._num_heads, self._attention_dim)
            delta = P_l * retain * v_base
            retain = retain * (1 - P_l)
            all_k_list.append(k_l)
            all_delta_list.append(delta)
            all_beta_list.append(beta_l)

        return all_k_list, all_delta_list, all_beta_list

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

        all_k_list, all_delta_list, all_beta_list = self._compute_solver_steps_with_k_base(
            solver_down_out, v_base, k_h
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


class PRISMr5Block(PRISMBlock):
    """
    PRISM Ablation r5 — ALL solver steps reuse K_base from main path.

    No independent K projections at all.
    Most extreme ablation: tests whether any independent key direction is needed.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # Remove solver_up_k entirely — all steps use K_base
        del self.solver_up_k_weight

    def _compute_solver_steps_with_k_base(self, solver_down_out, v_base, k_base_h):
        B, T = v_base.shape[0], v_base.shape[1]
        R = self._lr_rank
        S = self._solver_steps

        # Only need gain projections and gates
        gain_downs = solver_down_out[..., :R * S].chunk(S, dim=-1)
        gate_outs = solver_down_out[..., R * S * 2:].chunk(S, dim=-1)

        D = self.solver_up_gain_weight.shape[2]
        gain_input = torch.stack(gain_downs, dim=0).reshape(S, B * T, R)
        gain_out = torch.bmm(gain_input, self.solver_up_gain_weight)
        P_list = [torch.sigmoid(gain_out[i].view(B, T, D)) for i in range(S)]

        all_k_list, all_delta_list, all_beta_list = [], [], []
        retain = torch.ones_like(v_base)

        for step in range(S):
            P_l = P_list[step]
            beta_l = torch.sigmoid(gate_outs[step])
            delta = P_l * retain * v_base
            retain = retain * (1 - P_l)
            all_k_list.append(k_base_h)   # ALL steps reuse K_base
            all_delta_list.append(delta)
            all_beta_list.append(beta_l)

        return all_k_list, all_delta_list, all_beta_list

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

        all_k_list, all_delta_list, all_beta_list = self._compute_solver_steps_with_k_base(
            solver_down_out, v_base, k_h
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
