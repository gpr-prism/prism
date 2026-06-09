import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

class ATLAS(nn.Module):
    """
    ATLAS model
    """

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, 
                 device: str = 'cpu', polynomial_degree: int = 2, context_window: int = 4, memory_hidden_dim: int = None):
        """
        ATLAS model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param embedding_dim: int, dimension of embeddings
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        :param polynomial_degree: int, degree of polynomial kernel (increase basis complexity)
        :param context_window: int, context window size (kept but unused)
        :param memory_hidden_dim: int, hidden dimension (kept but unused)
        """
        super(ATLAS, self).__init__()

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
        self.polynomial_degree = polynomial_degree
        # context_window and memory_hidden_dim are kept but unused
        self.context_window = context_window
        self.memory_hidden_dim = memory_hidden_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.ID_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID' : nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        # Use AtlasBlock instead of SLABlock
        self.ATLAS_blocks = nn.ModuleList([
            AtlasBlock(
                embedding_dim=4*self.embedding_dim,
                linear_hidden_dim=4*self.embedding_dim,
                attention_dim=4*self.embedding_dim,
                dropout_ratio=self.dropout, 
                num_heads=self.num_heads, 
                max_seq_len=self.num_neighbors+1,
                chunk_size=self.num_neighbors+1,
                if_use_rope=True,
                epsilon=1e-6,
                polynomial_degree=self.polynomial_degree,
                context_window=self.context_window,
                memory_hidden_dim=self.memory_hidden_dim)
            for _ in range(self.num_layers)
        ])

        # Project to channel dimension (matches channel_embedding_dim) for downstream MergeLayer
        self.output_layer_src = nn.Linear(in_features=4*self.embedding_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2*self.embedding_dim, out_features=self.embedding_dim, bias=True)
    
    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 20):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        # pad the features of the sequence of source and destination nodes
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_nodes_neighbor_ID_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_neighbor_node_ids_torch = torch.from_numpy(src_neighbor_node_ids)
        src_nodes_neighbor_ID_features = self.projection_layer['ID'](src_nodes_neighbor_ID_features)

        src_node_features = [src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features,
                        src_nodes_neighbor_time_features, src_nodes_neighbor_ID_features]
        src_node_features = torch.concatenate(src_node_features, dim=-1)

        for ATLAS_blocks in self.ATLAS_blocks:
            # self-attention block with enhanced base complexity
            src_node_features = ATLAS_blocks(inputs=src_node_features, timestamps=src_neighbor_times)
        
        src_node_embeddings = self.output_layer_src(src_node_features[:, -1, :])

        return src_node_embeddings
    
    def compute_dst_node_temporal_embeddings(self, dst_node_ids: np.ndarray):
        # Tensor, shape (batch_size, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
        # Tensor, shape (batch_size, node_feat_dim)
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(dst_node_ids).to(self.device))
        
        # Tensor, shape (batch_size, node_feat_dim)
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
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float().to(self.device))
        # Tensor, shape (num_neighbors + 1, node_feat_dim)
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(nodes_neighbor_ids).to(self.device))

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features, nodes_neighbor_ID_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()


class AtlasBlock(nn.Module):
    """
    AtlasBlock - attention block with increased basis complexity
    """
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len, 
                 epsilon, chunk_size, if_use_rope, polynomial_degree=2, context_window=4, memory_hidden_dim=None):
        super(AtlasBlock, self).__init__()
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
        self._polynomial_degree = polynomial_degree

        # Network layers
        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))

        # Polynomial kernel projection layer (increase basis complexity)
        self._poly_query_proj = nn.Linear(self._attention_dim, self._attention_dim)
        self._poly_key_proj = nn.Linear(self._attention_dim, self._attention_dim)
        self._poly_value_proj = nn.Linear(self._linear_dim, self._linear_dim)

        # Layers from the original SLA
        self._uvqk = nn.Linear(self._embedding_dim, self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads * 1, self._embedding_dim * 5),
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads * 1)

        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)

        # RoPE
        pos = torch.arange(0, self._max_seq_len, dtype=torch.float32).unsqueeze(1)
        theta = torch.exp((-2 * math.log(10000) * torch.arange(0, self._attention_dim // 2, dtype=torch.float32) / self._attention_dim))
        vec = torch.stack([theta, theta], dim=-1).reshape(-1, self._attention_dim)
        rot = torch.tile(pos * vec, [1, self._num_heads])
        self.sin = torch.sin(rot)
        self.cos = torch.cos(rot)

    def _apply_rope(self, q, k):
        # Pre-reshape to (1, seq_len, num_heads, att_dim) to match q/k dimensions
        sin = self.sin.to(q.device).reshape(1, self._max_seq_len, self._num_heads, self._attention_dim)
        cos = self.cos.to(q.device).reshape(1, self._max_seq_len, self._num_heads, self._attention_dim)

        q_rot = torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1).reshape(q.shape)
        k_rot = torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1).reshape(k.shape)

        q = q * cos + q_rot * sin
        k = k * cos + k_rot * sin
        return q, k

    def _polynomial_kernel(self, x, degree):
        """
        Polynomial kernel feature mapping: increase basis complexity
        """
        result = x
        
        if degree >= 2:
            x_squared = x * x
            if x.shape[-1] == self._attention_dim:
                x_squared = self._poly_key_proj(x_squared)
            result = result + 0.3 * x_squared

        if degree >= 3:
            x_nonlinear = F.gelu(x) 
            if x.shape[-1] == self._attention_dim:
                x_nonlinear = self._poly_key_proj(x_nonlinear)
            result = result + 0.2 * x_nonlinear

        return result

    def _linear_attention_maybe_from_cache(self, q, k, v, invalid_attn_mask, initial_state):
        """
        Linear attention: use polynomial kernel while keeping SLA chunk-based structure
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        
        # Apply polynomial kernel to queries and keys
        q_poly = self._polynomial_kernel(q, self._polynomial_degree)
        k_poly = self._polynomial_kernel(k, self._polynomial_degree)
        
        # Keep SLA's chunk-based structure
        num_chunks = (seq_len + self._chunk_size - 1) // self._chunk_size  # ceil division
        
        # If seq_len is not divisible by chunk_size, pad
        padded_len = num_chunks * self._chunk_size
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            q_poly = F.pad(q_poly, (0, 0, 0, 0, 0, 0, 0, pad_size))
            k_poly = F.pad(k_poly, (0, 0, 0, 0, 0, 0, 0, pad_size))
            v = F.pad(v, (0, 0, 0, 0, 0, 0, 0, pad_size))
        
        q_chunks = q_poly.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        k_chunks = k_poly.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        v_chunks = v.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._linear_dim)
        
        # Initialize memory state
        memory_states = [initial_state]
        memory_state = memory_states[0]
        
        # Compute kv chunks (keys/values enhanced by polynomial kernel)
        k_de_v_chunks = torch.einsum('bnsha,bnshl->bnhal', k_chunks, v_chunks)
        
        # Accumulate memory state
        for i in range(num_chunks - 1):
            memory_state = memory_state + k_de_v_chunks[:, i]
            memory_states.append(memory_state)
        
        memory_states = torch.reshape(torch.cat(memory_states, dim=1), 
                                     [batch_size, num_chunks, self._num_heads, self._attention_dim, self._linear_dim])
        
        # Inter-chunk attention
        o_inter = torch.einsum('bnsha,bnhal->bnshl', q_chunks, memory_states)
        
        # Intra-chunk attention using polynomial kernel
        p = torch.einsum('bnshl, bnmhl->bnhsm', q_chunks, k_chunks)
        
        invalid_attn_mask_chunk = invalid_attn_mask[:self._chunk_size, :self._chunk_size]
        p_masked = p * invalid_attn_mask_chunk.to(p.device)
        
        o_intra = torch.einsum('bnhsm, bnmhl->bnshl', p_masked, v_chunks)
        
        # Combine inter- and intra-chunk attention
        o = o_inter + o_intra
        o = o.reshape(batch_size, padded_len, self._num_heads, self._linear_dim)
        
        # If padded, truncate back to original length
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
        
        # Reshape to multi-head format
        q = q.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        k = k.reshape(-1, self._max_seq_len, self._num_heads, self._attention_dim)
        v = v.reshape(-1, self._max_seq_len, self._num_heads, self._linear_dim)
        
        initial_state = torch.zeros((normed_x.shape[0], self._num_heads, self._attention_dim, self._linear_dim)).to(normed_x.device)

        if self._if_use_rope:
            q, k = self._apply_rope(q, k)

        attn_output = self._linear_attention_maybe_from_cache(q, k, v, self._invalid_attn_mask.to(normed_x.device), initial_state)
        
        g = self._g(normed_x)
        g = g.reshape(-1, self._max_seq_len, self._num_heads, self._linear_dim)
        g = nn.functional.silu(g)
        attn_output = g * attn_output

        attn_output = attn_output.reshape(-1, self._max_seq_len, self._num_heads * self._linear_dim)

        o_input = u * self.layer_norm_output(attn_output)

        new_x = self._o(
            self._dropout(o_input)
        ) + inputs

        return new_x
