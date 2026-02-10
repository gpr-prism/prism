"""
MAMBA-2: implemented on top of SLA, using the SSD framework with Euler discretization
1. Parallel parameter projections (A/B/C/Delta in parallel with input projection)
2. Multi-head SSM (consistent with multi-head attention structure)
3. Euler discretization (first-order, hardware-friendly)
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class MAMBA2(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 device: str = 'cpu', state_dim: int = 16, mimo_rank: int = 1):
        """
        MAMBA-2 model
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param embedding_dim: int, dimension of embeddings
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        :param state_dim: int, SSM state dimension (N)
        :param mimo_rank: int, MIMO rank (R), 1 for SISO, >1 for MIMO
        """
        super(MAMBA2, self).__init__()

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
        self.state_dim = state_dim
        self.mimo_rank = mimo_rank

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.ID_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID': nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        self.MAMBA2_blocks = nn.ModuleList([
            MAMBA2Block(
                embedding_dim=4 * self.embedding_dim,
                linear_hidden_dim=4 * self.embedding_dim,
                attention_dim=4 * self.embedding_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=self.num_neighbors + 1,
                chunk_size=self.num_neighbors + 1,
                if_use_rope=True,
                epsilon=1e-6,
                state_dim=self.state_dim,
                mimo_rank=self.mimo_rank)
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4 * self.embedding_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.embedding_dim, bias=True)

    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        """
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_ID_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_nodes_neighbor_ID_features = self.projection_layer['ID'](src_nodes_neighbor_ID_features)

        src_node_features = [src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features,
                             src_nodes_neighbor_time_features, src_nodes_neighbor_ID_features]
        src_node_features = torch.concatenate(src_node_features, dim=-1)

        for mamba2_blocks in self.MAMBA2_blocks:
            src_node_features = mamba2_blocks(inputs=src_node_features, timestamps=src_neighbor_times)

        src_node_embeddings = self.output_layer_src(src_node_features[:, -1, :])

        return src_node_embeddings

    def compute_dst_node_temporal_embeddings(self, dst_node_ids: np.ndarray):
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(dst_node_ids).to(self.device))

        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](nodes_neighbor_node_raw_features)
        dst_nodes_neighbor_ID_features = self.projection_layer['ID'](nodes_neighbor_ID_features)

        dst_node_features = [dst_nodes_neighbor_node_raw_features, dst_nodes_neighbor_ID_features]
        dst_node_features = torch.concatenate(dst_node_features, dim=-1)

        dst_node_embeddings = self.output_layer_dst(dst_node_features)

        return dst_node_embeddings

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge, time and depth features
        """
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(nodes_neighbor_ids).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_ID_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class MAMBA2Block(nn.Module):
    """
    MAMBA-2 Block - SSD-style SSM + SLA linear attention
    """
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len,
                 epsilon, chunk_size, if_use_rope, state_dim=16, mimo_rank=1):
        super(MAMBA2Block, self).__init__()
        # Params
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim // self._num_heads
        self._attention_dim = attention_dim // self._num_heads
        self._dropout_ratio = dropout_ratio
        self._max_seq_len = max_seq_len
        self._dropout = nn.Dropout(p=dropout_ratio)
        self._chunk_size = chunk_size
        self._if_use_rope = if_use_rope
        self._eps = epsilon
        self._state_dim = state_dim
        self._mimo_rank = mimo_rank

        # Network layers
        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))

        # SSM parameters: A, B, C, Delta
        self._A_log = nn.Parameter(torch.randn(self._num_heads, self._state_dim))
        self._B_proj = nn.Linear(self._embedding_dim, self._num_heads * self._state_dim * self._mimo_rank)
        self._C_proj = nn.Linear(self._embedding_dim, self._num_heads * self._state_dim * self._mimo_rank)
        self._delta_proj = nn.Linear(self._embedding_dim, self._num_heads * self._state_dim)

        # Input projection for MIMO SSM
        self._X_proj = nn.Linear(self._embedding_dim, self._num_heads * self._mimo_rank)

        # Standard SLA-like projections
        self._uvqk = nn.Linear(self._embedding_dim, self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads * self._mimo_rank, self._embedding_dim * 5),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads * self._mimo_rank)

        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads * self._mimo_rank)

        # Channel-specific biases
        self._B_bias = nn.Parameter(torch.zeros(self._num_heads, self._state_dim, self._mimo_rank))
        self._C_bias = nn.Parameter(torch.zeros(self._num_heads, self._state_dim, self._mimo_rank))

        # Projection layer for SSM output dimension matching
        self._ssm_output_proj = nn.Linear(self._num_heads * self._mimo_rank, self._num_heads * self._linear_dim)
        self._g_proj = nn.Linear(self._num_heads * self._linear_dim * self._mimo_rank, self._num_heads * self._linear_dim)

        # RoPE
        pos = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = torch.exp((-2 * math.log(10000) * torch.arange(0, self._attention_dim // 2, dtype=torch.float32) / self._attention_dim))
        vec = torch.stack([theta, theta], dim=-1).reshape(-1, self._attention_dim)
        rot = torch.tile(pos * vec, [1, self._num_heads])
        self.sin = torch.sin(rot)
        self.cos = torch.cos(rot)

    def _apply_rope(self, q, k):
        self.sin = self.sin.to(q.device)
        self.cos = self.cos.to(q.device)
        q_rot = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1)
        k_rot = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1)
        q_rot = q_rot.reshape(q.shape)
        k_rot = k_rot.reshape(k.shape)
        q = q * self.cos + q_rot * self.sin
        k = k * self.cos + k_rot * self.sin

        return q, k

    def _euler_discretization(self, A, B_t, delta_t, x_t, h_t_prev):
        """
        Euler discretization (first-order)
        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * x_t
        """
        delta_t_flat = delta_t.squeeze(1) if delta_t.dim() > 3 else delta_t  # [B, H, N]
        A_expanded = A.unsqueeze(0)  # [1, H, N]
        alpha_t = torch.exp(delta_t_flat * A_expanded)  # [B, H, N]

        alpha_t = alpha_t.unsqueeze(-1)  # [B, H, N, 1]
        delta_t_expanded = delta_t_flat.unsqueeze(-1)  # [B, H, N, 1]

        h_t = alpha_t * h_t_prev
        Bx_t = torch.einsum('bhnr,bhr->bhn', B_t, x_t)  # [B, H, N]
        Bx_t = Bx_t.unsqueeze(-1).expand(-1, -1, -1, self._mimo_rank)  # [B, H, N, R]
        h_t = h_t + delta_t_expanded * Bx_t

        return h_t

    def _mamba2_ssm_forward(self, x, normed_x, delta):
        """
        MAMBA-2 SSM forward pass (Euler discretization)
        :param x: Tensor, shape (batch_size, seq_len, num_heads, mimo_rank) - MIMO input
        :param normed_x: Tensor, shape (batch_size, seq_len, embedding_dim) - normalized input for B/C projection
        :param delta: Tensor, shape (batch_size, seq_len, num_heads, state_dim) - time steps
        :return: output Tensor, shape (batch_size, seq_len, num_heads * mimo_rank)
        """
        batch_size, seq_len, num_heads, mimo_rank = x.shape

        A = -torch.exp(self._A_log)  # [H, N]

        B_flat = self._B_proj(normed_x)  # [B, L, H*N*R]
        B = B_flat.reshape(batch_size, seq_len, num_heads, self._state_dim, mimo_rank)
        B = B + self._B_bias.unsqueeze(0).unsqueeze(0)

        C_flat = self._C_proj(normed_x)  # [B, L, H*N*R]
        C = C_flat.reshape(batch_size, seq_len, num_heads, self._state_dim, mimo_rank)
        C = C + self._C_bias.unsqueeze(0).unsqueeze(0)

        h = torch.zeros(batch_size, num_heads, self._state_dim, mimo_rank, device=normed_x.device)
        outputs = []

        for t in range(seq_len):
            delta_t = delta[:, t, :, :]  # [B, H, N]
            B_t = B[:, t, :, :, :]       # [B, H, N, R]
            C_t = C[:, t, :, :, :]       # [B, H, N, R]
            x_t = x[:, t, :, :]          # [B, H, R]

            h = self._euler_discretization(A, B_t, delta_t, x_t, h)
            y_t = torch.einsum('bhnr,bhnr->bhr', C_t, h)  # [B, H, R]
            outputs.append(y_t)

        output = torch.stack(outputs, dim=1)  # [B, L, H, R]
        output = output.reshape(batch_size, seq_len, num_heads * mimo_rank)

        return output

    def _linear_attention_maybe_from_cache(self, q, k, v, invalid_attn_mask, initial_state):
        """
        Keep SLA's chunk-based linear attention structure
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        num_chunks = (seq_len + self._chunk_size - 1) // self._chunk_size
        padded_len = num_chunks * self._chunk_size

        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            q = F.pad(q, (0, 0, 0, 0, 0, 0, 0, pad_size))
            k = F.pad(k, (0, 0, 0, 0, 0, 0, 0, pad_size))
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad_size))

        q_chunks = q.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        k_chunks = k.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        v_chunks = v.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._linear_dim)

        memory_states = [initial_state]
        memory_state = memory_states[0]

        k_de_v_chunks = torch.einsum('bnsha,bnshl->bnhal', k_chunks, v_chunks)

        for i in range(num_chunks - 1):
            memory_state += k_de_v_chunks[:, i]
            memory_states.append(memory_state)

        memory_states = torch.reshape(torch.cat(memory_states, dim=1),
                                      [batch_size, num_chunks, self._num_heads, self._attention_dim, self._linear_dim])

        o_inter = torch.einsum('bnsha,bnhal->bnshl', q_chunks, memory_states)

        p = torch.einsum('bnshl, bnmhl->bnhsm', q_chunks, k_chunks)
        invalid_attn_mask_chunk = invalid_attn_mask[:self._chunk_size, :self._chunk_size]
        p_masked = p * invalid_attn_mask_chunk.to(p.device)

        o_intra = torch.einsum('bnhsm, bnmhl->bnshl', p_masked, v_chunks)

        o = o_inter + o_intra
        o = o.reshape(batch_size, padded_len, self._num_heads, self._linear_dim)

        if padded_len > seq_len:
            o = o[:, :seq_len, :, :]

        return o

    def forward(self, inputs, timestamps):
        """
        :param inputs: Tensor, shape (batch_size, num_neighbors + 1, embedding_dim)
        :param timestamps: Tensor, shape (batch_size, num_neighbors + 1)
        """
        normed_x = self.layer_norm_input(inputs)
        batched_mm_output = self._uvqk(normed_x)

        u, v, q, k = torch.split(
            batched_mm_output,
            [
                self._linear_dim * self._num_heads,
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads
            ],
            dim=-1
        )

        q = q.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        k = k.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        v = v.reshape(-1, self._max_seq_len, self._num_heads, self._linear_dim)

        initial_state = torch.zeros((normed_x.shape[0], self._num_heads, self._attention_dim, self._linear_dim)).to(normed_x.device)

        attn_output = self._linear_attention_maybe_from_cache(q, k, v, self._invalid_attn_mask.to(normed_x.device), initial_state)

        # MAMBA-2 SSM processing (parallel path)
        x_mimo = self._X_proj(normed_x)  # [B, L, H*R]
        x_mimo = x_mimo.reshape(-1, self._max_seq_len, self._num_heads, self._mimo_rank)

        delta = self._delta_proj(normed_x)  # [B, L, H*N]
        delta = delta.reshape(-1, self._max_seq_len, self._num_heads, self._state_dim)
        delta = F.softplus(delta)

        ssm_output = self._mamba2_ssm_forward(x_mimo, normed_x, delta)  # [B, L, H*R]
        ssm_output = ssm_output.reshape(-1, self._max_seq_len, self._num_heads, self._mimo_rank)

        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]

        attn_output_flat = attn_output.reshape(batch_size, seq_len, self._num_heads * self._linear_dim)
        ssm_output_flat = ssm_output.reshape(batch_size, seq_len, self._num_heads * self._mimo_rank)
        ssm_output_expanded = self._ssm_output_proj(ssm_output_flat)

        combined_output = 0.7 * attn_output_flat + 0.3 * ssm_output_expanded

        g = self._g(normed_x)
        g = g.reshape(batch_size, seq_len, self._linear_dim * self._num_heads * self._mimo_rank)
        g = self._g_proj(g)
        g = F.silu(g)
        combined_output = g * combined_output

        u_reshaped = u.reshape(batch_size, seq_len, self._num_heads * self._linear_dim)
        o_input = u_reshaped * self.layer_norm_output(combined_output)

        new_x = self._o(
            self._dropout(o_input)
        ) + inputs

        return new_x
