import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler

class HSTU(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler, num_neighbors: int,
                 time_feat_dim: int, embedding_dim: int, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1, device: str = 'cpu'):
        """
        HSTU model.
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
        super(HSTU, self).__init__()

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

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            'ID' : nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        self.HSTU_blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim=4*self.embedding_dim,
                linear_hidden_dim=4*self.embedding_dim,  # 1 -> 2 -> 4
                attention_dim=4*self.embedding_dim, # 1 -> 2 -> 4
                dropout_ratio=self.dropout, 
                num_heads=self.num_heads, 
                max_seq_len=self.num_neighbors+1,
                if_use_rope=True,
                epsilon=1e-6)
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=4*self.embedding_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2*self.embedding_dim, out_features=self.embedding_dim, bias=True)
        #self.output_layer_dst2 = nn.Linear(self.embedding_dim, self.node_feat_dim)
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

        for HSTU_blocks in self.HSTU_blocks:
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = HSTU_blocks(inputs=src_node_features, timestamps=src_neighbor_times)
        
        # src_node_features = torch.mean(src_node_features, dim=1)
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


class HSTUBlock(nn.Module):
    def __init__(self, embedding_dim, linear_hidden_dim, attention_dim, dropout_ratio, num_heads, max_seq_len, if_use_rope, epsilon):
        super(HSTUBlock, self).__init__()
        # Params
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim
        self._linear_dim = linear_hidden_dim // self._num_heads
        self._attention_dim = attention_dim // self._num_heads
        self._dropout_ratio = dropout_ratio
        self._max_seq_len = max_seq_len
        self._dropout = nn.Dropout(p=dropout_ratio)
        self._if_use_rope = if_use_rope
        self._eps = epsilon

        # Network layers
        self._rel_attn_bias = RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=self._max_seq_len, 
            num_bucketed=32, 
            bucketization_fn=TimeIntervalBucketFn
        )
        self._invalid_attn_mask = torch.tril(torch.ones(self._max_seq_len, self._max_seq_len))

        self._uvqk = nn.Linear(self._embedding_dim, self._linear_dim * 2 * self._num_heads + self._attention_dim * self._num_heads * 2)  # 4 -> 2
        self._o = nn.Sequential(
            nn.Linear(self._linear_dim * self._num_heads * 1, self._embedding_dim * 5),  # 3 -> 1
            nn.SiLU(),
            nn.Linear(self._embedding_dim * 5, self._embedding_dim),
        )

        self.layer_norm_input = nn.LayerNorm(self._embedding_dim)
        self.layer_norm_output = nn.LayerNorm(self._linear_dim * self._num_heads * 1)  # 3 -> 1

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

    def _hstu_attention_maybe_from_cache(self, q, k, v, timestamps, invalid_attn_mask):
        n = q.shape[1]
        qk_attn = torch.einsum(
            'bnhd,bmhd->bhnm',
            q.reshape(-1, n, self._num_heads, self._linear_dim),
            k.reshape(-1, n, self._num_heads, self._linear_dim),
        )
        qk_attn = F.silu(qk_attn) / n
        invalid_attn_mask = invalid_attn_mask.unsqueeze(0)
        qk_attn *= invalid_attn_mask.unsqueeze(0)

        pos_attn, ts_attn = self._rel_attn_bias(timestamps, device=q.device)
        pos_attn *= invalid_attn_mask
        ts_attn *= invalid_attn_mask

        qk_attn = F.normalize(qk_attn, p=2, dim=-1)
        pos_attn = F.normalize(pos_attn, p=2, dim=-1)
        ts_attn = F.normalize(ts_attn, p=2, dim=-1)

        output_latent = torch.einsum(
            'bhnm,bmhd->bnhd', 
            qk_attn, 
            v.reshape(-1, n, self._num_heads, self._linear_dim)
        )
        output_pos = torch.einsum(
            'bnm,bmhd->bnhd', 
            pos_attn, 
            v.reshape(-1, n, self._num_heads, self._linear_dim)
        )
        output_ts = torch.einsum(
            'bnm,bmhd->bnhd', 
            ts_attn, 
            v.reshape(-1, n, self._num_heads, self._linear_dim)
        )

        # combined_output = torch.concat([output_latent, output_pos, output_ts], axis=-1)  # (batch_size, n, self._num_heads, self._linear_dim * 3)
        # output_latent = combined_output.reshape(-1, n, self._num_heads * self._linear_dim * 3)
        output_latent = output_latent.reshape(-1, n, self._num_heads * self._linear_dim)
        return output_latent
    
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

        if self._if_use_rope:
            q, k = self._apply_rope(q, k)

        self._invalid_attn_mask = self._invalid_attn_mask.to(q.device)
        attn_output = self._hstu_attention_maybe_from_cache(q, k, v, timestamps, self._invalid_attn_mask)

        o_input = u * self.layer_norm_output(attn_output)

        new_x = self._o(
            self._dropout(o_input)
        ) + inputs

        return new_x


class RelativeBucketedTimeAndPositionBasedBias(nn.Module):
    def __init__(self, max_seq_len, num_bucketed, bucketization_fn):
        super(RelativeBucketedTimeAndPositionBasedBias, self).__init__()
        self._max_seq_len = max_seq_len
        self._num_buckets = num_bucketed
        self._bucketization_fn = bucketization_fn
        
        self._ts_w = nn.Parameter(torch.randn(self._num_buckets + 1) * 0.02)
        self._pos_w = nn.Parameter(torch.randn(2 * self._max_seq_len - 1) * 0.02)

    def forward(self, all_timestamps, device):
        all_timestamps = torch.from_numpy(all_timestamps).to(device)

        B = all_timestamps.shape[0]
        N = self._max_seq_len
        t = F.pad(self._pos_w[: 2 * N - 1], [0, N])  # (3 * N - 1, )
        t = torch.tile(t, [N])  # (N, 3 * N - 1)
        t = torch.reshape(t[..., :-N], (1, N, 3 * N - 2))  # (1, N, 3 * N - 2)
        r = (2 * N - 1) // 2

        # [B, N + 1] to simplify tensor manipulations.
        ext_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1 : N]], dim=1)  # (B, N + 1)
        # causal masking. Otherwise [:, :-1] - [:, 1:] works
        bucketed_timestamps = torch.clamp(
            self._bucketization_fn(
                torch.unsqueeze(ext_timestamps[:, 1:], 2) - torch.unsqueeze(ext_timestamps[:, :-1], 1)
            ),
            min=0,
            max=self._num_buckets,
        )

        rel_pos_bias = t[:, :, r:-r]  # (1, N, N)
        rel_pos_bias = rel_pos_bias.clone()
        rel_ts_bias = torch.gather(self._ts_w * 1.0, 0, torch.reshape(bucketed_timestamps, [-1]))
        rel_ts_bias = rel_ts_bias.clone()
        rel_ts_bias = torch.reshape(rel_ts_bias, [-1, N, N])

        return torch.tile(rel_pos_bias, [B, 1, 1]), rel_ts_bias


def TimeIntervalBucketFn(x):
    x_clip = torch.clamp(torch.abs(x), min=1, max=torch.iinfo(torch.int64).max)
    return (torch.log(x_clip.to(dtype=torch.float32)) / 0.031).to(dtype=torch.int64)
