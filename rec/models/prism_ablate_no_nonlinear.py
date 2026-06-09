"""
PRISM (Ablation RQ3): w/o Non-Linearity
Paper ablation: remove non-linearity inside the solver (GELU/SiLU) and use a purely linear update.

1) Write-Forget Decoupling
   S_t = A_t S_{t-1} + B_t                               (Eq. 8)
   A_t: linear forgetting (state-independent, parallel prefix-scan)
   B_t: high-rank write (Rank-L injection)
   B_t = \\sum_{l=1}^{L} \\beta_t^{(l)} (\\delta_t^{(l)} \\otimes k_t^{(l)})  (Eq. 10)

2) Input-Anchored Loop Unrolling
   u_t = ShortConv(X_{<=t}) \\approx S_{t-1} k_t          (Eq. 9)
   r^{(1)} = v_t - u_t                                   (Eq. 14)
   p_t^{(l)} = W_p^{(l)} u_t \\approx \\sigma'(S_{t-1}k_t) (Eq. 13)
   for l = 1..L:
       \\delta_t^{(l)} = GELU(p_t^{(l)} \\odot r_t^{(l)}) (Eq. 15)
       r_t^{(l+1)} = r_t^{(l)} - \\delta_t^{(l)}          (Eq. 16)
       Note: in this ablation, Eq. 15 uses a linear map instead of a non-linearity.
       k_t^{(l)} = W_k^{(l)} u_t                          (Eq. 12)

3) Rank-L accumulation: B_t is the parallel sum of L rank-1 outer products
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class PRISM(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 num_neighbors: int, time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, device: str = 'cpu', solver_steps: int = 3, short_kernel_size: int = 5,
                 gate_low_rank_dim: int = 16, gate_temperature: float = 16.0, gate_min: float = 1e-4,
                 injection_init: float = -2.0, solver_out_init: float = -2.0):
        """
        PRISM model (ablation: w/o Non-Linearity)
        - Forget path: linear, state-independent recurrence A_t (parallel prefix-scan; Eq. 7/8)
        - Write path: input-anchored + Rank-L iterative injection B_t (Eq. 9–18)
        """
        super(PRISM, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.num_nodes = self.node_raw_features.shape[0]
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.num_neighbors = num_neighbors
        self.time_feat_dim = time_feat_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.solver_steps = max(1, int(solver_steps))
        self.short_kernel_size = max(3, int(short_kernel_size))
        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_temperature = gate_temperature
        self.gate_min = gate_min
        self.injection_init = injection_init
        self.solver_out_init = solver_out_init

        # Time encoder and ID embedding
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.ID_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID': nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        # Stacked layers: each block has a linear recurrence backbone (A_t) + PRISM injection branch (B_t)
        self.PRISM_blocks = nn.ModuleList([
            PRISMBlock(
                embedding_dim=4 * self.embedding_dim,
                linear_hidden_dim=4 * self.embedding_dim,
                attention_dim=4 * self.embedding_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=self.num_neighbors + 1,
                chunk_size=self.num_neighbors + 1,
                if_use_rope=True,
                epsilon=1e-6,
                solver_steps=self.solver_steps,
                short_kernel_size=self.short_kernel_size,
                gate_low_rank_dim=self.gate_low_rank_dim,
                gate_temperature=self.gate_temperature,
                gate_min=self.gate_min,
                injection_init=self.injection_init,
                solver_out_init=self.solver_out_init,
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4 * self.embedding_dim, out_features=self.node_feat_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, num_neighbors: int = 20):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong),
                                                src_neighbor_edge_ids), axis=1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
            src_nodes_neighbor_ID_features = self.get_features(node_interact_times=node_interact_times,
                                                               nodes_neighbor_ids=src_neighbor_node_ids,
                                                               nodes_edge_ids=src_neighbor_edge_ids,
                                                               nodes_neighbor_times=src_neighbor_times,
                                                               time_encoder=self.time_encoder)

        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_nodes_neighbor_ID_features = self.projection_layer['ID'](src_nodes_neighbor_ID_features)

        # x_t = [node, edge, time, id] as input sequence features X
        src_node_features = [src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features,
                             src_nodes_neighbor_time_features, src_nodes_neighbor_ID_features]
        src_node_features = torch.cat(src_node_features, dim=-1)

        for prism_block in self.PRISM_blocks:
            # Sequence update block corresponding to "input-anchored + Rank-L injection" in the paper
            src_node_features = prism_block(inputs=src_node_features, timestamps=src_neighbor_times)

        src_node_embeddings = self.output_layer_src(src_node_features[:, -1, :])
        return src_node_embeddings

    def compute_dst_node_temporal_embeddings(self, dst_node_ids: np.ndarray):
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(dst_node_ids).to(self.device))

        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](nodes_neighbor_node_raw_features)
        dst_nodes_neighbor_ID_features = self.projection_layer['ID'](nodes_neighbor_ID_features)

        dst_node_features = [dst_nodes_neighbor_node_raw_features, dst_nodes_neighbor_ID_features]
        dst_node_features = torch.cat(dst_node_features, dim=-1)
        dst_node_embeddings = self.output_layer_dst(dst_node_features)
        return dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(
            timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device)
        )
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(nodes_neighbor_ids).to(self.device))
        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_ID_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class PRISMBlock(nn.Module):
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads,
                 max_seq_len, epsilon, chunk_size, if_use_rope, solver_steps: int, short_kernel_size: int,
                 gate_low_rank_dim: int, gate_temperature: float, gate_min: float,
                 injection_init: float, solver_out_init: float):
        super(PRISMBlock, self).__init__()
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim // self._num_heads
        self._attention_dim = attention_dim // self._num_heads
        self._dropout_ratio = dropout_ratio
        self._max_seq_len = max_seq_len
        self._chunk_size = chunk_size
        self._if_use_rope = if_use_rope
        self._eps = epsilon
        self._solver_steps = max(1, int(solver_steps))
        self._short_kernel_size = max(3, int(short_kernel_size))
        self._gate_low_rank_dim = gate_low_rank_dim
        self._gate_temperature = gate_temperature
        self._gate_min = gate_min

        self._dropout = nn.Dropout(p=dropout_ratio)
        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))

        # Linear attention backbone (forget path): state-independent data-dependent decay
        # Corresponds to A_t in the paper (Eq. 7/8), enabling parallel prefix-scan
        self._uvqk = nn.Linear(self._embedding_dim,
                               self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads, self._embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 4, self._embedding_dim),
        )
        # Forget gate g_t: implements A_t = diag(g_t) as data-dependent decay
        # This is the forget term in Write-Forget Decoupling, kept state-independent for parallel scan
        self._gate = nn.Sequential(
            nn.Linear(self._embedding_dim, self._gate_low_rank_dim),
            nn.Linear(self._gate_low_rank_dim, self._attention_dim * self._num_heads),
            nn.Sigmoid(),
        )
        # Gate mapping: convert forget-gate statistics into write-branch scale factors
        self._gk_to_embed = nn.Linear(self._attention_dim, self._embedding_dim)
        self._gk_to_heads = nn.Linear(self._attention_dim, self._num_heads)
        self._gk_to_inject = nn.Sequential(
            nn.Linear(self._attention_dim, self._attention_dim),
            nn.SiLU(),
            nn.Linear(self._attention_dim, 1),
        )
        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads)
        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)

        # PRISM solver branch (write path)
        # Stage-1: input-anchored proxy (ShortConv), u_t = ShortConv(X_{<=t})  (Eq. 9)
        self.short_conv = nn.Conv1d(self._embedding_dim, self._embedding_dim,
                                    kernel_size=self._short_kernel_size, groups=self._embedding_dim, bias=True)
        self.anchor_norm = nn.LayerNorm(self._embedding_dim)

        # Contextual gain predictor: approximates p_t^{(l)} or σ' term (Eq. 13)
        self.gain_mlp = nn.Sequential(
            nn.Linear(self._embedding_dim, self._embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 2, self._embedding_dim),
        )
        # Residual initialization: learnable approximation of r^{(1)} = v_t - u_t (Eq. 14)
        self.residual_proj = nn.Linear(self._embedding_dim, self._embedding_dim)
        self.residual_norm = nn.LayerNorm(self._embedding_dim)

        # Stage-2: iterative refinement (Rank-L accumulation), corresponds to Eq. 10–16
        self.step_k_proj = nn.ModuleList([nn.Linear(self._embedding_dim, self._attention_dim * self._num_heads)
                                          for _ in range(self._solver_steps)])
        self.step_v_proj = nn.ModuleList([nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)
                                          for _ in range(self._solver_steps)])
        self.step_gate = nn.ModuleList([nn.Linear(self._embedding_dim, self._num_heads)
                                        for _ in range(self._solver_steps)])
        # Ablation: remove solver non-linearity, use a purely linear map for updates
        self.step_delta = nn.ModuleList([nn.Linear(self._embedding_dim, self._embedding_dim)
                                         for _ in range(self._solver_steps)])

        # Aggregated injection output (parallel residual injection into representation space)
        # Maps the "visible updates" of Rank-L injection back to the representation space
        self.solver_out = nn.Sequential(
            nn.LayerNorm(self._embedding_dim),
            nn.Linear(self._embedding_dim, self._embedding_dim),
        )

        # Learnable scaling: controls injection strength and residual branch strength
        # (aligned with the overall B_t scale in Eq. 18)
        self._injection_logit = nn.Parameter(torch.tensor(float(injection_init)))
        self._solver_out_logit = nn.Parameter(torch.tensor(float(solver_out_init)))

        # RoPE
        pos = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = torch.exp((-2 * math.log(10000) * torch.arange(0, self._attention_dim // 2, dtype=torch.float32)
                           / self._attention_dim))
        vec = torch.stack([theta, theta], dim=-1).reshape(-1, self._attention_dim)
        rot = torch.tile(pos * vec, [1, self._num_heads])
        self.sin = torch.sin(rot)
        self.cos = torch.cos(rot)

    def _apply_rope(self, q, k):
        self.sin = self.sin.to(q.device)
        self.cos = self.cos.to(q.device)
        q_rot = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1).reshape(q.shape)
        k_rot = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1).reshape(k.shape)
        q = q * self.cos + q_rot * self.sin
        k = k * self.cos + k_rot * self.sin
        return q, k

    def _short_conv_anchor(self, x):
        x_t = x.transpose(1, 2)  # (B, D, T)
        x_pad = F.pad(x_t, (self._short_kernel_size - 1, 0))
        anchor = self.short_conv(x_pad)[..., :x_t.size(-1)]
        return anchor.transpose(1, 2)

    def _compute_prism_injection(self, anchor, gain, gk_scale, gk_head_gate, gk_inject_gate):
        """
        Compute Rank-L injection term (write path):
        B_t = \\sum_{l=1}^{L} \\beta_t^{(l)} (\\delta_t^{(l)} \\otimes k_t^{(l)})  (Eq. 10)
        """
        batch_size, seq_len, _ = anchor.shape
        # Learnable approximation of r^{(1)} (implementation of Eq. 14)
        residual = self.residual_norm(self.residual_proj(anchor))

        injection_k_list = []
        injection_v_list = []
        update_embed = torch.zeros_like(residual)

        for step in range(self._solver_steps):
            # \\beta_t^{(l)}: per-head gate (scalar), implementing the gate coefficient in Eq. 11
            head_gate = torch.sigmoid(self.step_gate[step](anchor))  # (B, T, H)
            if gk_head_gate is not None:
                head_gate = head_gate * (1.0 + gk_head_gate)
            gate_scalar = head_gate.mean(dim=2, keepdim=True)  # (B, T, 1)

            # Implementation site of \\delta_t^{(l)} (Eq. 15)
            # Ablation: replace GELU/SiLU with a linear map to test non-linearity importance
            # Here gain is the shared proxy for p_t^{(l)}, residual is r_t^{(l)}
            delta = gate_scalar * (gain * residual)
            if gk_scale is not None:
                # Scale modulation for training stability; does not change Rank-L structure
                delta = delta * (1.0 + gk_scale)
            if gk_inject_gate is not None:
                delta = delta * (1.0 + gk_inject_gate)
            delta = self.step_delta[step](delta)

            # Accumulate visible residual updates (for the output branch)
            update_embed = update_embed + delta

            # r^{(l+1)} = r^{(l)} - \\delta_t^{(l)}  (Eq. 16)
            residual = residual - delta

            # k_t^{(l)} = W_k^{(l)} u_t  (Eq. 12)
            k_l = self.step_k_proj[step](anchor).view(batch_size, seq_len, self._num_heads, self._attention_dim)
            # v_t^{(l)}: linear projection of \\delta_t^{(l)} (for outer-product writing)
            v_l = self.step_v_proj[step](delta).view(batch_size, seq_len, self._num_heads, self._linear_dim)
            v_l = v_l * head_gate.unsqueeze(-1)  # head-level gating (equivalent to \\beta_t^{(l)})
            if gk_head_gate is not None:
                k_l = k_l * (1.0 + gk_head_gate.unsqueeze(-1))
                v_l = v_l * (1.0 + gk_head_gate.unsqueeze(-1))
            if gk_inject_gate is not None:
                v_l = v_l * (1.0 + gk_inject_gate.unsqueeze(-1))

            injection_k_list.append(k_l)
            injection_v_list.append(v_l)

        return injection_k_list, injection_v_list, update_embed

    def _gated_linear_attention_with_injection(self, q, k, v, gk, injection_k_list, injection_v_list, gk_inject_gate,
                                               invalid_attn_mask, initial_state):
        batch_size = q.shape[0]
        num_chunks = self._max_seq_len // self._chunk_size
        q_chunks = q.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        k_chunks = k.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        v_chunks = v.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._linear_dim)
        gk_chunks = gk.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)

        # Forgetting (state-independent): A_t = diag(g_t)  (Eq. 7/8)
        decay_start = torch.cumprod(gk_chunks, dim=2)
        decay_end = torch.flip(torch.cumprod(torch.flip(gk_chunks, dims=[2]), dim=2), dims=[2])
        prod_gk = torch.prod(gk_chunks, dim=2)
        ones = torch.ones(size=[self._linear_dim], device=gk_chunks.device)
        gkv = torch.einsum('bnha,l->bnhal', prod_gk, ones)

        # Base write (Rank-1 outer product): the single-step update of linear attention/recurrence
        k_de = k_chunks * decay_end
        k_de_v_chunks = torch.einsum('bnsha,bnshl->bnhal', k_de, v_chunks)

        # Rank-L injection (from PRISM solver): corresponds to B_t in Eq. 10/18
        injection_kv_chunks = None
        if injection_k_list is not None and len(injection_k_list) > 0:
            injection_kv_chunks = torch.zeros_like(k_de_v_chunks)
            for k_l, v_l in zip(injection_k_list, injection_v_list):
                k_l_chunks = k_l.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
                v_l_chunks = v_l.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._linear_dim)
                k_l_de = k_l_chunks * decay_end
                injection_kv_chunks = injection_kv_chunks + torch.einsum('bnsha,bnshl->bnhal', k_l_de, v_l_chunks)

        # Learnable scaling for injection strength (aligned with B_t scale in Eq. 18)
        injection_scale = torch.sigmoid(self._injection_logit)
        injection_scale_chunks = injection_scale
        gk_inject_chunks = None
        gk_inject_chunk_mean = None
        if gk_inject_gate is not None:
            gk_inject_chunks = gk_inject_gate.reshape(batch_size, num_chunks, self._chunk_size, 1, 1)
            gk_inject_chunk_mean = gk_inject_chunks.mean(dim=2)
            injection_scale_chunks = injection_scale * (1.0 + gk_inject_chunk_mean)

        memory_states = [initial_state]
        memory_state = memory_states[0]
        for i in range(num_chunks - 1):
            if injection_kv_chunks is not None:
                if gk_inject_chunk_mean is not None:
                    injection_chunk = injection_kv_chunks[:, i] * (1.0 + gk_inject_chunk_mean[:, i])
                    scale = injection_scale_chunks[:, i]
                    memory_state = memory_state * gkv[:, i] + k_de_v_chunks[:, i] + scale * injection_chunk
                else:
                    memory_state = memory_state * gkv[:, i] + k_de_v_chunks[:, i] + injection_scale_chunks * injection_kv_chunks[:, i]
            else:
                memory_state = memory_state * gkv[:, i] + k_de_v_chunks[:, i]
            memory_states.append(memory_state)

        memory_states = torch.reshape(torch.cat(memory_states, dim=1),
                                      [batch_size, num_chunks, self._num_heads, self._attention_dim, self._linear_dim])

        # Attention output (parallel prefix scan + intra-chunk attention)
        q_ds = q_chunks * decay_start
        o_inter = torch.einsum('bnsha,bnhal->bnshl', q_ds, memory_states)

        k_ds = k_chunks / (decay_start + self._eps)
        p = torch.einsum('bnshl,bnmhl->bnhsm', q_ds, k_ds)
        invalid_attn_mask_chunk = invalid_attn_mask[:self._chunk_size, :self._chunk_size]
        p_masked = p * invalid_attn_mask_chunk
        o_intra = torch.einsum('bnhsm,bnmhl->bnshl', p_masked, v_chunks)

        o = o_inter + o_intra
        o = o.reshape(batch_size, self._max_seq_len, self._num_heads, self._linear_dim)
        return o

    def forward(self, inputs, timestamps):
        normed_x = self.layer_norm_input(inputs)

        # Linear attention backbone: forget path (A_t)
        batched_mm_output = self._uvqk(normed_x)
        u, v, q, k = torch.split(
            batched_mm_output,
            [self._linear_dim * self._num_heads,
             self._linear_dim * self._num_heads,
             self._attention_dim * self._num_heads,
             self._attention_dim * self._num_heads],
            dim=-1
        )
        initial_state = torch.zeros((normed_x.shape[0], self._num_heads, self._attention_dim, self._linear_dim),
                                    device=normed_x.device)

        # Compute forget gate g_t (data-dependent but state-independent, parallel-scan compliant)
        gk = self._gate(normed_x)
        if self._gate_temperature != 1.0:
            gk = gk ** (1.0 / self._gate_temperature)
        gk = gk.clamp(min=self._gate_min, max=1.0)

        # Use forget-gate statistics as write-branch scales (implementation-level stabilization/calibration)
        gk_token = gk.reshape(normed_x.shape[0], self._max_seq_len, self._num_heads, self._attention_dim).mean(dim=2)
        gk_scale = torch.sigmoid(self._gk_to_embed(gk_token))
        gk_head_gate = torch.sigmoid(self._gk_to_heads(gk_token))
        gk_inject_gate = torch.sigmoid(self._gk_to_inject(gk_token))

        if self._if_use_rope:
            q, k = self._apply_rope(q, k)

        # PRISM write path: input anchoring + Rank-L injection (Eq. 9–18)
        anchor = self.anchor_norm(self._short_conv_anchor(normed_x))
        # Inject scale into anchor and gain
        anchor = anchor * (1.0 + gk_scale)
        gain = torch.sigmoid(self.gain_mlp(anchor))
        gain = gain * (1.0 + gk_scale)
        injection_k_list, injection_v_list, update_embed = self._compute_prism_injection(
            anchor, gain, gk_scale, gk_head_gate, gk_inject_gate
        )

        attn_output = self._gated_linear_attention_with_injection(
            q, k, v, gk, injection_k_list, injection_v_list, gk_inject_gate,
            self._invalid_attn_mask.to(normed_x.device), initial_state
        )
        g = self._g(normed_x).reshape(-1, self._max_seq_len, self._num_heads, self._linear_dim)
        g = F.silu(g)
        attn_output = g * attn_output
        attn_output = attn_output.reshape(-1, self._max_seq_len, self._num_heads * self._linear_dim)
        attn_output = self._o(self._dropout(u * self.layer_norm_output(attn_output)))

        # Injection output branch (explicit residual injection; effective even for a single chunk)
        # Implements B_t contribution in Eq. 18 as a parallel residual path
        solver_out_scale = torch.sigmoid(self._solver_out_logit)
        solver_out = self.solver_out(update_embed) * solver_out_scale
        solver_out = solver_out * (1.0 + gk_inject_gate)

        # Final output: linear recurrence backbone + Rank-L injection branch (Write-Forget Decoupling)
        new_x = inputs + self._dropout(attn_output) + self._dropout(solver_out)
        return new_x
