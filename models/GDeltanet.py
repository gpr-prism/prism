import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


import fla  # Top-level package

# Import specific components
from myfla.gated_deltanet import GatedDeltaNet

class GDeltanet(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        """
        GDeltanet model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param embedding_dim: int, dimension of embeddings
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(GDeltanet, self).__init__()

        #self.node_raw_features = nn.Parameter(torch.from_numpy(node_raw_features.astype(np.float32)), requires_grad = True).to(device)
        #self.edge_raw_features = nn.Parameter(torch.from_numpy(edge_raw_features.astype(np.float32)), requires_grad = True).to(device)

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

        #print(f'node_feat_dim is:{self.node_feat_dim}')
        
        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID' : nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        self.GDeltanet_blocks = nn.ModuleList([
            GatedDeltaNet(
                hidden_size = 4*self.embedding_dim,
                expand_v = 1, # linear dim
                num_heads = self.num_heads,
                head_dim = max(32, (4 * self.embedding_dim) // max(1, self.num_heads) // 2),
                mode = 'fused_recurrent',
                force_fallback = True,
                use_rope = True,
                max_seq_len = self.num_neighbors+1
            )#.to(dtype=torch.bfloat16)
            for _ in range(self.num_layers)
        ])
        '''
        self.SLA_blocks = nn.ModuleList([
            SLABlock(
                embedding_dim=4*self.embedding_dim,
                linear_hidden_dim=4*self.embedding_dim,  # 1 -> 2 -> 4
                attention_dim=4*self.embedding_dim, # 1 -> 2 -> 4
                dropout_ratio=self.dropout, 
                num_heads=self.num_heads, 
                max_seq_len=self.num_neighbors+1,
                chunk_size=self.num_neighbors+1,
                if_use_rope=True,
                epsilon=1e-6)
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4*self.embedding_dim, out_features=self.node_feat_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2*self.embedding_dim, out_features=self.node_feat_dim, bias=True)
        '''
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
        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors)
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
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
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

        # manually adding residual connection between layers
        residual = torch.zeros_like(src_node_features)
        # attention mask ([batch size, seq_len], for speeding up)
        # Use edge ids to detect padding (0 means padding/self edge at position 0).
        src_neighbor_edge_ids_torch = torch.from_numpy(src_neighbor_edge_ids).to(src_node_features.device)
        mask = src_neighbor_edge_ids_torch != 0
        mask[:, 0] = True
        #mask  = torch.ones_like(mask)
        #mask = mask.to(torch.bfloat16)
        #src_node_features = src_node_features.to(torch.bfloat16)
        residual = residual.to(torch.bfloat16)

        for GDeltanet_block in self.GDeltanet_blocks:
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = src_node_features + residual
            src_node_features = GDeltanet_block(hidden_states=src_node_features, attention_mask=mask)[0]
            residual = src_node_features
        
        #src_node_features = src_node_features.to(torch.float32)
        # src_node_features = torch.mean(src_node_features, dim=1)
        src_node_embeddings = self.output_layer_src(src_node_features[:, -1, :])
        
        #print(f'la block output shape is: {src_node_embeddings.shape}')
        
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
    def get_real_token_count(self, src_node_ids: np.ndarray,node_interact_times: np.ndarray, num_neighbors: int = 20):
        '''
        get the real token number (for load balancing)
        :param src_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return: real token number [batch size, 1]
        '''
        real_token_counts = self.neighbor_sampler.get_real_token_counts(node_ids=src_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        return real_token_counts


'''
class SLABlock(nn.Module):
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len, epsilon, chunk_size, if_use_rope):
        super(SLABlock, self).__init__()
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

        # Network layers
        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))

        self._uvqk = nn.Linear(self._embedding_dim, self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)  # 4 -> 2
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads * 1, self._embedding_dim * 5),  # 3 -> 1
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads * 1)  # 3 -> 1

        self._g = nn.Linear(self._embedding_dim, self._linear_dim * self._num_heads)

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

    def _linear_attention_maybe_from_cache(self, q, k, v, invalid_attn_mask, initial_state):
        batch_size = q.shape[0]
        
        num_chunks = self._max_seq_len // self._chunk_size
        q_chunks = q.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        k_chunks = k.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)
        v_chunks = v.reshape(batch_size, num_chunks, self._chunk_size, self._num_heads, self._attention_dim)

        # memory state initialization
        memory_states = [initial_state]
        memory_state = memory_states[0]

        # kv chunks
        k_de_v_chunks = torch.einsum(
            'bnsha,bnshl->bnhal',
            k_chunks,
            v_chunks
        )
        
        for i in range(num_chunks - 1):
            memory_state += k_de_v_chunks[:, i]
            memory_states.append(memory_state)

        memory_states = torch.reshape(torch.cat(memory_states, dim=1), [batch_size, num_chunks, self._num_heads, self._attention_dim, self._linear_dim])

        o_inter = torch.einsum(
            'bnsha,bnhal->bnshl',
            q_chunks,
            memory_states
        )
        
        
        p = torch.einsum(
            'bnshl, bnmhl->bnhsm',
            q_chunks,
            k_chunks
        )

        invalid_attn_mask_chunk = invalid_attn_mask[:self._chunk_size, :self._chunk_size]
        p_masked = p * invalid_attn_mask_chunk

        o_intra = torch.einsum(
            'bnhsm, bnmhl->bnshl',
            p_masked,
            v_chunks
        )

        o = o_inter + o_intra
        o = o.reshape(batch_size, self._max_seq_len, self._num_heads, self._linear_dim)

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
                self._linear_dim * self._num_heads,  # remove * 3
                self._linear_dim * self._num_heads,
                self._attention_dim * self._num_heads,
                self._attention_dim * self._num_heads
            ], 
            dim=-1
        )
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

'''
