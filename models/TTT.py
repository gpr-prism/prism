"""
TTT: SLA structure + TTT-Linear
- per-token inner-loop update of W,b (online linear regression)
- prediction target uses residual form (v - k)
- learnable TTT LR gate + token index scaling
- RoPE + normalization, output dimension matches linear_dim exactly
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


class TTT(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 device: str = 'cpu'):
        super(TTT, self).__init__()

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

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.ID_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID': nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        self.TTT_blocks = nn.ModuleList([
            TTTBlock(
                embedding_dim=4 * self.embedding_dim,
                linear_hidden_dim=4 * self.embedding_dim,
                attention_dim=4 * self.embedding_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=self.num_neighbors + 1,
                chunk_size=self.num_neighbors + 1,
                if_use_rope=True,
                epsilon=1e-6)
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4 * self.embedding_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.embedding_dim, bias=True)

    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, num_neighbors: int = 20):
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

        for ttt_block in self.TTT_blocks:
            src_node_features = ttt_block(inputs=src_node_features, timestamps=src_neighbor_times)

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
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(nodes_neighbor_ids).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_ID_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class TTTBlock(nn.Module):
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len,
                 epsilon, chunk_size, if_use_rope, eta_base: float = 1.0):
        super(TTTBlock, self).__init__()
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
        self._eta_base = eta_base

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads)
        self.ttt_norm = nn.LayerNorm(self._attention_dim)

        self._q_proj = nn.Linear(self._embedding_dim, self._num_heads * self._attention_dim)
        self._k_proj = nn.Linear(self._embedding_dim, self._num_heads * self._attention_dim)
        self._v_proj = nn.Linear(self._embedding_dim, self._num_heads * self._attention_dim)
        self._u = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)
        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)
        self._lr_proj = nn.Linear(self._embedding_dim, self._num_heads)

        self._theta_k = nn.Linear(self._attention_dim, self._attention_dim)
        self._theta_q = nn.Linear(self._attention_dim, self._attention_dim)
        self._theta_v = nn.Linear(self._attention_dim, self._attention_dim)

        # Learnable initialization of W0 and b0 per head
        self._W0 = nn.Parameter(torch.zeros(self._num_heads, self._attention_dim, self._attention_dim))
        self._b0 = nn.Parameter(torch.zeros(self._num_heads, 1, self._attention_dim))

        self._ttt_out_proj = nn.Linear(self._attention_dim, self._linear_dim)

        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads, self._embedding_dim * 5),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        # RoPE
        pos = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = torch.exp((-2 * math.log(10000) * torch.arange(0, self._attention_dim // 2, dtype=torch.float32) / self._attention_dim))
        vec = torch.stack([theta, theta], dim=-1).reshape(-1, self._attention_dim)
        rot = pos * vec
        self.sin = torch.sin(rot)
        self.cos = torch.cos(rot)

        # token_idx (1 / idx) with learnable offset
        token_idx = 1.0 / torch.arange(1, self._max_seq_len + 1, dtype=torch.float32)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(self._max_seq_len))

    def _apply_rope(self, q, k):
        sin = self.sin.to(q.device).unsqueeze(0).unsqueeze(2)
        cos = self.cos.to(q.device).unsqueeze(0).unsqueeze(2)
        q_rot = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1)
        k_rot = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1)
        q_rot = q_rot.reshape(q.shape)
        k_rot = k_rot.reshape(k.shape)
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

    def _get_eta(self, normed_x):
        lr_gate = torch.sigmoid(self._lr_proj(normed_x))  # [B, L, H]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = torch.clamp(token_idx, min=0.0)
        eta = (self._eta_base * token_idx).view(1, -1, 1) * lr_gate
        eta = eta / math.sqrt(self._attention_dim)
        return eta

    def _ttt_forward(self, q, k, v, eta):
        batch_size, seq_len, num_heads, dim = q.shape
        W = self._W0.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        b = self._b0.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        outputs = []

        for t in range(seq_len):
            k_t = self._theta_k(k[:, t])
            v_t = self._theta_v(v[:, t])
            q_t = self._theta_q(q[:, t])

            z_t = torch.einsum('bhdm,bhm->bhd', W, k_t) + b.squeeze(2)
            z_t = self.ttt_norm(z_t)
            ssl_target = v_t - k_t
            err = z_t - ssl_target

            grad = torch.einsum('bhd,bhm->bhdm', err, k_t)
            eta_t = eta[:, t].unsqueeze(-1).unsqueeze(-1)
            W = W - eta_t * grad
            b = b - eta_t * err.unsqueeze(2)

            out_t = q_t + z_t
            outputs.append(out_t)

        output = torch.stack(outputs, dim=1)  # [B, L, H, D_attn]
        return output

    def forward(self, inputs, timestamps):
        normed_x = self.layer_norm_input(inputs)
        q = self._q_proj(normed_x)
        k = self._k_proj(normed_x)
        v = self._v_proj(normed_x)

        q = q.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        k = k.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        v = v.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)

        q = F.normalize(q, dim=-1, eps=self._eps)
        k = F.normalize(k, dim=-1, eps=self._eps)
        v = F.normalize(v, dim=-1, eps=self._eps)

        if self._if_use_rope:
            q, k = self._apply_rope(q, k)

        eta = self._get_eta(normed_x)
        attn_output = self._ttt_forward(q, k, v, eta)
        attn_output = self._ttt_out_proj(attn_output)

        attn_output = attn_output.reshape(-1, self._max_seq_len, self._num_heads * self._linear_dim)
        g = self._g(normed_x)
        g = F.silu(g)
        attn_output = g * attn_output

        u = self._u(normed_x)
        o_input = u * self.layer_norm_output(attn_output)

        new_x = self._o(self._dropout(o_input)) + inputs
        return new_x
