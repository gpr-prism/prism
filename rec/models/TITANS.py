import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

_TITANS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "titans-pytorch-main"))
if os.path.isdir(_TITANS_ROOT) and _TITANS_ROOT not in sys.path:
    sys.path.append(_TITANS_ROOT)

try:
    from titans_pytorch.neural_memory import NeuralMemory
    _NEURAL_MEMORY_AVAILABLE = True
except Exception:
    NeuralMemory = None
    _NEURAL_MEMORY_AVAILABLE = False


class TITANS(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1,
                 device: str = 'cpu'):
        """
        TITANS model (SLA short-term + Titans neural memory).
        """
        super(TITANS, self).__init__()

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

        self.TITANS_blocks = nn.ModuleList([
            TITANSBlock(
                embedding_dim=4 * self.embedding_dim,
                linear_hidden_dim=4 * self.embedding_dim,
                attention_dim=4 * self.embedding_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=self.num_neighbors + 1,
                chunk_size=self.num_neighbors + 1,
                if_use_rope=True,
                epsilon=1e-6,
                use_neural_memory=_NEURAL_MEMORY_AVAILABLE
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4 * self.embedding_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.embedding_dim, bias=True)

    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                             node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source node temporal embeddings
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

        for titans_block in self.TITANS_blocks:
            src_node_features = titans_block(inputs=src_node_features, timestamps=src_neighbor_times)

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


class NeuralMemoryLite(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        self.theta_k = nn.Linear(self.head_dim, self.head_dim)
        self.theta_q = nn.Linear(self.head_dim, self.head_dim)
        self.theta_v = nn.Linear(self.head_dim, self.head_dim)

        self.eta_proj = nn.Linear(embedding_dim, num_heads)
        self.theta_proj = nn.Linear(embedding_dim, num_heads)
        self.alpha_proj = nn.Linear(embedding_dim, num_heads)

        self.W0 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.head_dim))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        eta = torch.sigmoid(self.eta_proj(x)).clamp(1e-3, 1.0 - 1e-3)
        theta = torch.sigmoid(self.theta_proj(x)).clamp(1e-3, 1.0 - 1e-3)
        alpha = torch.sigmoid(self.alpha_proj(x)).clamp(1e-3, 1.0 - 1e-3)

        W = self.W0.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        S = torch.zeros_like(W)
        outputs = []
        scale = 1.0 / math.sqrt(self.head_dim)

        for t in range(seq_len):
            k_t = self.theta_k(k[:, t])
            v_t = self.theta_v(v[:, t])
            q_t = self.theta_q(q[:, t])

            pred = torch.einsum('bhdm,bhm->bhd', W, k_t)
            err = pred - v_t
            grad = 2.0 * torch.einsum('bhd,bhm->bhdm', err, k_t) * scale

            eta_t = eta[:, t].unsqueeze(-1).unsqueeze(-1)
            theta_t = theta[:, t].unsqueeze(-1).unsqueeze(-1)
            alpha_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = theta_t * S + grad
            W = (1.0 - alpha_t) * W - eta_t * S

            out_t = torch.einsum('bhdm,bhm->bhd', W, q_t)
            outputs.append(out_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs.reshape(batch_size, seq_len, self.num_heads * self.head_dim)


class TITANSBlock(nn.Module):
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len,
                 epsilon, chunk_size, if_use_rope, use_neural_memory: bool = True):
        super(TITANSBlock, self).__init__()
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim // self._num_heads
        self._attention_dim = attention_dim // self._num_heads
        self._max_seq_len = max_seq_len
        self._dropout = nn.Dropout(p=dropout_ratio)
        self._chunk_size = chunk_size
        self._if_use_rope = if_use_rope
        self._eps = epsilon

        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))
        self._uvqk = nn.Linear(self._embedding_dim, self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)

        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads, self._embedding_dim * 5),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads)
        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)

        pos = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = torch.exp((-2 * math.log(10000) * torch.arange(0, self._attention_dim // 2, dtype=torch.float32) / self._attention_dim))
        vec = torch.stack([theta, theta], dim=-1).reshape(-1, self._attention_dim)
        rot = torch.tile(pos * vec, [1, self._num_heads])
        self.sin = torch.sin(rot)
        self.cos = torch.cos(rot)

        self.mem_norm = nn.LayerNorm(self._embedding_dim)
        self.mem_gate = nn.Linear(self._embedding_dim, 1)
        self.mem_out_proj = nn.Linear(self._embedding_dim, self._embedding_dim)

        self.use_neural_memory = use_neural_memory and _NEURAL_MEMORY_AVAILABLE
        if self.use_neural_memory:
            self.neural_memory = NeuralMemory(
                dim=self._embedding_dim,
                chunk_size=self._chunk_size,
                dim_head=self._attention_dim,
                heads=self._num_heads,
                activation=nn.SiLU(),
                momentum=True,
                qk_rmsnorm=True,
                max_grad_norm=2.0
            )
        else:
            self.neural_memory = NeuralMemoryLite(self._embedding_dim, self._num_heads)

    def _apply_rope(self, q, k):
        sin = self.sin.to(q.device)
        cos = self.cos.to(q.device)
        q_rot = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1)
        k_rot = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1)
        q_rot = q_rot.reshape(q.shape)
        k_rot = k_rot.reshape(k.shape)
        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

    def _linear_attention_maybe_from_cache(self, q, k, v, invalid_attn_mask, initial_state):
        batch_size = q.shape[0]
        num_chunks = self._max_seq_len // self._chunk_size
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
        invalid_attn_mask_chunk = invalid_attn_mask[:self._chunk_size, :self._chunk_size].to(p.device)
        p_masked = p * invalid_attn_mask_chunk
        o_intra = torch.einsum('bnhsm, bnmhl->bnshl', p_masked, v_chunks)

        o = o_inter + o_intra
        o = o.reshape(batch_size, self._max_seq_len, self._num_heads, self._linear_dim)
        return o

    def forward(self, inputs, timestamps):
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

        initial_state = torch.zeros((normed_x.shape[0], self._num_heads, self._attention_dim, self._linear_dim), device=normed_x.device)

        if self._if_use_rope:
            q, k = self._apply_rope(q, k)

        attn_output = self._linear_attention_maybe_from_cache(q, k, v, self._invalid_attn_mask, initial_state)
        g = self._g(normed_x)
        g = g.reshape(-1, self._max_seq_len, self._num_heads, self._linear_dim)
        g = nn.functional.silu(g)
        attn_output = g * attn_output

        attn_output = attn_output.reshape(-1, self._max_seq_len, self._num_heads * self._linear_dim)
        o_input = u * self.layer_norm_output(attn_output)
        short_out = self._o(self._dropout(o_input)) + inputs

        mem_in = self.mem_norm(inputs)
        if self.use_neural_memory:
            mem_out, _ = self.neural_memory(mem_in)
        else:
            mem_out = self.neural_memory(mem_in)

        mem_gate = torch.sigmoid(self.mem_gate(mem_in))
        mem_out = self.mem_out_proj(mem_out)
        fused = short_out + self._dropout(mem_out * mem_gate)

        return fused
