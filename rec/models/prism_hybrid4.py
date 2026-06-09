import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.HSTU import HSTUBlock
from models.modules import TimeEncoder
from models.prism import PRISMBlock
from utils.utils import NeighborSampler


class PRISMMoMBlock(nn.Module):
    """
    MoM-routed PRISM layer with dedicated experts + shared expert(s).
    """

    def __init__(
        self,
        embedding_dim: int,
        linear_hidden_dim: int,
        attention_dim: int,
        dropout_ratio: float,
        num_heads: int,
        max_seq_len: int,
        chunk_size: int,
        if_use_rope: bool,
        epsilon: float,
        solver_steps: int,
        short_kernel_size: int,
        gate_low_rank_dim: int,
        gate_temperature: float,
        gate_min: float,
        injection_init: float,
        solver_out_init: float,
        num_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        router_temperature: float = 1.0,
        router_jitter_eps: float = 0.01,
    ):
        super(PRISMMoMBlock, self).__init__()
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if num_shared_experts <= 0:
            raise ValueError(f"num_shared_experts must be positive, got {num_shared_experts}")

        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.router_temperature = max(1e-6, float(router_temperature))
        self.router_jitter_eps = max(0.0, float(router_jitter_eps))

        self.router = nn.Linear(embedding_dim, self.num_experts)
        self.shared_logit = nn.Parameter(torch.tensor(-1.0))
        self.dropout = nn.Dropout(p=dropout_ratio)

        def build_prism_block():
            return PRISMBlock(
                embedding_dim=embedding_dim,
                linear_hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                dropout_ratio=dropout_ratio,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                chunk_size=chunk_size,
                if_use_rope=if_use_rope,
                epsilon=epsilon,
                solver_steps=solver_steps,
                short_kernel_size=short_kernel_size,
                gate_low_rank_dim=gate_low_rank_dim,
                gate_temperature=gate_temperature,
                gate_min=gate_min,
                injection_init=injection_init,
                solver_out_init=solver_out_init,
            )

        self.experts = nn.ModuleList([build_prism_block() for _ in range(self.num_experts)])
        self.shared_experts = nn.ModuleList([build_prism_block() for _ in range(self.num_shared_experts)])

    def forward(self, inputs, timestamps):
        router_logits = self.router(inputs)
        if self.training and self.router_jitter_eps > 0.0:
            router_logits = router_logits + self.router_jitter_eps * torch.randn_like(router_logits)
        router_probs = F.softmax(router_logits / self.router_temperature, dim=-1)

        topk_weights, selected_experts = torch.topk(router_probs, k=self.top_k, dim=-1, sorted=True)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)
        one_hot = F.one_hot(selected_experts, num_classes=self.num_experts).to(dtype=inputs.dtype)
        routing_weights = torch.einsum("btk,btke->bte", topk_weights, one_hot)

        mixed_delta = torch.zeros_like(inputs)
        for expert_idx, expert in enumerate(self.experts):
            expert_out = expert(inputs=inputs, timestamps=timestamps)
            expert_delta = expert_out - inputs
            mixed_delta = mixed_delta + routing_weights[:, :, expert_idx:expert_idx + 1] * expert_delta

        shared_delta = torch.zeros_like(inputs)
        for expert in self.shared_experts:
            shared_delta = shared_delta + (expert(inputs=inputs, timestamps=timestamps) - inputs)
        shared_delta = shared_delta / float(self.num_shared_experts)

        shared_scale = torch.sigmoid(self.shared_logit)
        return inputs + self.dropout(mixed_delta + shared_scale * shared_delta)


class PRISMHybrid4(nn.Module):
    """
    Four-layer hybrid architecture:
    1) HSTU
    2) PRISM + MoM (4 routed experts + 1 shared expert)
    3) PRISM + MoM (4 routed experts + 1 shared expert)
    4) HSTU
    """

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        num_neighbors: int,
        time_feat_dim: int,
        embedding_dim: int,
        num_layers: int = 4,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        solver_steps: int = 3,
        short_kernel_size: int = 5,
        gate_low_rank_dim: int = 16,
        gate_temperature: float = 16.0,
        gate_min: float = 1e-4,
        injection_init: float = -2.0,
        solver_out_init: float = -2.0,
        num_experts: int = 4,
        num_shared_experts: int = 1,
        top_k: int = 2,
        router_temperature: float = 1.0,
        router_jitter_eps: float = 0.01,
    ):
        super(PRISMHybrid4, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.num_nodes = self.node_raw_features.shape[0]
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.num_neighbors = num_neighbors
        self.time_feat_dim = time_feat_dim
        self.embedding_dim = embedding_dim
        self.num_layers = 4  # fixed architecture
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.ID_embedding = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.embedding_dim)

        self.projection_layer = nn.ModuleDict({
            "node": nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            "edge": nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            "time": nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            "ID": nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        hidden_dim = 4 * self.embedding_dim
        seq_len = self.num_neighbors + 1

        self.hybrid_blocks = nn.ModuleList([
            HSTUBlock(
                embedding_dim=hidden_dim,
                linear_hidden_dim=hidden_dim,
                attention_dim=hidden_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=seq_len,
                if_use_rope=True,
                epsilon=1e-6,
            ),
            PRISMMoMBlock(
                embedding_dim=hidden_dim,
                linear_hidden_dim=hidden_dim,
                attention_dim=hidden_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=seq_len,
                chunk_size=seq_len,
                if_use_rope=True,
                epsilon=1e-6,
                solver_steps=max(1, int(solver_steps)),
                short_kernel_size=max(3, int(short_kernel_size)),
                gate_low_rank_dim=gate_low_rank_dim,
                gate_temperature=gate_temperature,
                gate_min=gate_min,
                injection_init=injection_init,
                solver_out_init=solver_out_init,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                router_temperature=router_temperature,
                router_jitter_eps=router_jitter_eps,
            ),
            PRISMMoMBlock(
                embedding_dim=hidden_dim,
                linear_hidden_dim=hidden_dim,
                attention_dim=hidden_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=seq_len,
                chunk_size=seq_len,
                if_use_rope=True,
                epsilon=1e-6,
                solver_steps=max(1, int(solver_steps)),
                short_kernel_size=max(3, int(short_kernel_size)),
                gate_low_rank_dim=gate_low_rank_dim,
                gate_temperature=gate_temperature,
                gate_min=gate_min,
                injection_init=injection_init,
                solver_out_init=solver_out_init,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                router_temperature=router_temperature,
                router_jitter_eps=router_jitter_eps,
            ),
            HSTUBlock(
                embedding_dim=hidden_dim,
                linear_hidden_dim=hidden_dim,
                attention_dim=hidden_dim,
                dropout_ratio=self.dropout,
                num_heads=self.num_heads,
                max_seq_len=seq_len,
                if_use_rope=True,
                epsilon=1e-6,
            ),
        ])

        self.output_layer_src = nn.Linear(in_features=4 * self.embedding_dim, out_features=self.node_feat_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_node_temporal_embeddings(self, src_node_ids: np.ndarray, node_interact_times: np.ndarray,
                                             num_neighbors: int = 20):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            node_ids=src_node_ids,
            node_interact_times=node_interact_times,
            num_neighbors=num_neighbors,
        )

        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        src_neighbor_edge_ids = np.concatenate(
            (np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1
        )
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, \
            src_nodes_neighbor_ID_features = self.get_features(
                node_interact_times=node_interact_times,
                nodes_neighbor_ids=src_neighbor_node_ids,
                nodes_edge_ids=src_neighbor_edge_ids,
                nodes_neighbor_times=src_neighbor_times,
                time_encoder=self.time_encoder,
            )

        src_nodes_neighbor_node_raw_features = self.projection_layer["node"](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer["edge"](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer["time"](src_nodes_neighbor_time_features)
        src_nodes_neighbor_ID_features = self.projection_layer["ID"](src_nodes_neighbor_ID_features)

        src_node_features = torch.cat([
            src_nodes_neighbor_node_raw_features,
            src_nodes_edge_raw_features,
            src_nodes_neighbor_time_features,
            src_nodes_neighbor_ID_features,
        ], dim=-1)

        for hybrid_block in self.hybrid_blocks:
            src_node_features = hybrid_block(inputs=src_node_features, timestamps=src_neighbor_times)

        src_node_embeddings = self.output_layer_src(src_node_features[:, -1, :])
        return src_node_embeddings

    def compute_dst_node_temporal_embeddings(self, dst_node_ids: np.ndarray):
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(dst_node_ids).to(self.device))

        dst_nodes_neighbor_node_raw_features = self.projection_layer["node"](nodes_neighbor_node_raw_features)
        dst_nodes_neighbor_ID_features = self.projection_layer["ID"](nodes_neighbor_ID_features)

        dst_node_features = torch.cat([dst_nodes_neighbor_node_raw_features, dst_nodes_neighbor_ID_features], dim=-1)
        return self.output_layer_dst(dst_node_features)

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
        if self.neighbor_sampler.sample_neighbor_strategy in ["uniform", "time_interval_aware"]:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()
