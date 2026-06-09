"""
TTTv2: graph-task wrapper around the official chunk/minibatch TTT-Linear block.

This implementation mirrors the official `ttt-lm-jax` TTT-Linear structure:
- depthwise causal pre-conv
- RMSNorm -> chunked TTT-Linear -> residual
- RMSNorm -> SwiGLU MLP -> residual
- minibatch/chunk-wise hidden-state updates with the official lower-triangular
  closed-form write/read equations
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import TimeEncoder
from utils.utils import NeighborSampler


def _init_linear(linear: nn.Linear, std: float = 0.02) -> nn.Linear:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)
    return linear


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout_ratio: float, initializer_range: float = 0.02):
        super(SwiGLUMLP, self).__init__()
        self.w1 = _init_linear(nn.Linear(hidden_size, intermediate_size, bias=False), std=initializer_range)
        self.w2 = _init_linear(nn.Linear(intermediate_size, hidden_size, bias=False), std=initializer_range)
        self.w3 = _init_linear(nn.Linear(hidden_size, intermediate_size, bias=False), std=initializer_range)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        return self.dropout(x)


class DepthwiseCausalConv1d(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, initializer_range: float = 0.02):
        super(DepthwiseCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=hidden_size,
            bias=True,
        )
        nn.init.normal_(self.conv.weight, mean=0.0, std=initializer_range)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        y = self.conv(x.transpose(1, 2))
        y = y[..., :seq_len]
        return y.transpose(1, 2).contiguous()


class HeadwiseLayerNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6):
        super(HeadwiseLayerNorm, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads, head_dim))

    def _affine(self, x: torch.Tensor):
        extra_dims = x.dim() - 3
        view_shape = [1, self.num_heads] + [1] * extra_dims + [self.head_dim]
        return self.weight.view(*view_shape), self.bias.view(*view_shape)

    def _norm_stats(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        inv_std = torch.rsqrt(centered.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_hat = centered * inv_std
        return x_hat, inv_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self._norm_stats(x)
        weight, bias = self._affine(x)
        return x_hat * weight + bias

    def vjp_input(self, x: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
        x_hat, inv_std = self._norm_stats(x)
        weight, _ = self._affine(x)
        dx_hat = grad_output * weight
        dim = x.shape[-1]
        sum_dx_hat = dx_hat.sum(dim=-1, keepdim=True)
        sum_dx_hat_x_hat = (dx_hat * x_hat).sum(dim=-1, keepdim=True)
        return (dx_hat * dim - sum_dx_hat - x_hat * sum_dx_hat_x_hat) * (inv_std / dim)


class ConvModule(nn.Module):
    def __init__(self, hidden_size: int, conv_width: int, eps: float = 1e-6, initializer_range: float = 0.02):
        super(ConvModule, self).__init__()
        self.conv_norm = RMSNorm(hidden_size, eps=eps)
        self.conv = DepthwiseCausalConv1d(hidden_size=hidden_size, kernel_size=conv_width, initializer_range=initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(self.conv_norm(hidden_states))


class TTTLinear(nn.Module):
    """
    PyTorch port of the official chunk/minibatch TTT-Linear layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mini_batch_size: int,
        conv_width: int = 4,
        rope_theta: float = 10000.0,
        ttt_base_lr: float = 1.0,
        eps: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        super(TTTLinear, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        if mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")

        self.mini_batch_size = mini_batch_size
        self.rope_theta = rope_theta
        self.ttt_base_lr = ttt_base_lr

        self.wq = _init_linear(nn.Linear(hidden_size, hidden_size, bias=False), std=initializer_range)
        self.wv = _init_linear(nn.Linear(hidden_size, hidden_size, bias=False), std=initializer_range)
        self.wo = _init_linear(nn.Linear(hidden_size, hidden_size, bias=False), std=initializer_range)
        self.wg = _init_linear(nn.Linear(hidden_size, hidden_size, bias=False), std=initializer_range)
        self.learnable_ttt_lr = _init_linear(nn.Linear(hidden_size, num_heads, bias=True), std=initializer_range)

        self.conv_q = DepthwiseCausalConv1d(hidden_size=hidden_size, kernel_size=conv_width, initializer_range=initializer_range)
        self.conv_k = DepthwiseCausalConv1d(hidden_size=hidden_size, kernel_size=conv_width, initializer_range=initializer_range)

        self.ttt_norm = HeadwiseLayerNorm(num_heads=num_heads, head_dim=self.head_dim, eps=eps)
        self.post_norm = nn.LayerNorm(hidden_size, eps=eps)

        self.W1 = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        self.b1 = nn.Parameter(torch.zeros(num_heads, 1, self.head_dim))
        nn.init.normal_(self.W1, mean=0.0, std=initializer_range)

        token_idx = 1.0 / torch.arange(1, mini_batch_size + 1, dtype=torch.float32)
        self.register_buffer("token_idx", token_idx, persistent=False)
        self.learnable_token_idx = nn.Parameter(torch.zeros(mini_batch_size))

        positions = torch.arange(0, mini_batch_size * 2, dtype=torch.float32).unsqueeze(1)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        freqs = positions * inv_freq.unsqueeze(0)
        rope = torch.stack([freqs, freqs], dim=-1).reshape(positions.shape[0], self.head_dim)
        self.register_buffer("rope_sin", torch.sin(rope), persistent=False)
        self.register_buffer("rope_cos", torch.cos(rope), persistent=False)

    def _split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        return hidden_states.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

    def _apply_rope(self, xq: torch.Tensor, xk: torch.Tensor, position_ids: torch.Tensor):
        rope_idx = position_ids % self.mini_batch_size
        sin = self.rope_sin.to(xq.device)[rope_idx].unsqueeze(2)
        cos = self.rope_cos.to(xq.device)[rope_idx].unsqueeze(2)

        xq_rot = torch.stack([-xq[..., 1::2], xq[..., 0::2]], dim=-1).reshape_as(xq)
        xk_rot = torch.stack([-xk[..., 1::2], xk[..., 0::2]], dim=-1).reshape_as(xk)
        xq = xq * cos + xq_rot * sin
        xk = xk * cos + xk_rot * sin
        return xq, xk

    def _get_eta(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        lr_gate = torch.sigmoid(self.learnable_ttt_lr(hidden_states))
        token_idx = torch.clamp(self.token_idx + self.learnable_token_idx, min=0.0)
        eta_scale = token_idx[(position_ids % self.mini_batch_size).long()].unsqueeze(-1)
        eta = self.ttt_base_lr * eta_scale * lr_gate / float(self.head_dim)
        return eta

    def _process_mini_batch(
        self,
        xq_mini_batch: torch.Tensor,
        xk_mini_batch: torch.Tensor,
        xv_mini_batch: torch.Tensor,
        eta_mini_batch: torch.Tensor,
        W1_init: torch.Tensor,
        b1_init: torch.Tensor,
    ):
        x1 = xk_mini_batch
        z1 = torch.einsum("bhmd,bhdf->bhmf", x1, W1_init) + b1_init
        ttt_norm_out = self.ttt_norm(z1)

        ssl_target = xv_mini_batch - xk_mini_batch
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target
        grad_l_wrt_z1 = self.ttt_norm.vjp_input(z1, grad_l_wrt_ttt_norm_out)

        x1_bar = xq_mini_batch
        mini_batch_size = xq_mini_batch.shape[2]
        tril_mask = torch.tril(torch.ones(mini_batch_size, mini_batch_size, device=xq_mini_batch.device, dtype=xq_mini_batch.dtype))
        attn1 = torch.einsum("bhid,bhjd->bhij", x1_bar, x1) * tril_mask
        eta_rows = eta_mini_batch.unsqueeze(-1)
        eta_tril = eta_rows * tril_mask.view(1, 1, mini_batch_size, mini_batch_size)

        b1_bar = b1_init - torch.matmul(eta_tril, grad_l_wrt_z1)
        z1_bar = torch.einsum("bhmd,bhdf->bhmf", x1_bar, W1_init) - torch.matmul(eta_rows * attn1, grad_l_wrt_z1) + b1_bar
        ttt_norm_out_bar = self.ttt_norm(z1_bar)

        output_mini_batch = x1_bar + ttt_norm_out_bar

        last_eta = eta_mini_batch[:, :, -1:].unsqueeze(-1)
        W1_bar_last = W1_init - torch.einsum("bhmd,bhmf->bhdf", last_eta * x1, grad_l_wrt_z1)
        b1_bar_last = b1_init - torch.sum(last_eta * grad_l_wrt_z1, dim=2, keepdim=True)

        return W1_bar_last, b1_bar_last, output_mini_batch

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        ttt_lr_mult: float = 1.0,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)

        xqk = self.wq(hidden_states)
        xv = self.wv(hidden_states)
        xq = self.conv_q(xqk)
        xk = self.conv_k(xqk)

        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)
        xq, xk = self._apply_rope(xq, xk, position_ids=position_ids)

        eta = self._get_eta(hidden_states, position_ids=position_ids) * ttt_lr_mult

        W1 = self.W1.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
        b1 = self.b1.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

        outputs = []
        for start in range(0, seq_len, self.mini_batch_size):
            end = min(start + self.mini_batch_size, seq_len)
            xq_mini_batch = xq[:, start:end].transpose(1, 2).contiguous()
            xk_mini_batch = xk[:, start:end].transpose(1, 2).contiguous()
            xv_mini_batch = xv[:, start:end].transpose(1, 2).contiguous()
            eta_mini_batch = eta[:, start:end].transpose(1, 2).contiguous()

            W1, b1, output_mini_batch = self._process_mini_batch(
                xq_mini_batch=xq_mini_batch,
                xk_mini_batch=xk_mini_batch,
                xv_mini_batch=xv_mini_batch,
                eta_mini_batch=eta_mini_batch,
                W1_init=W1,
                b1_init=b1,
            )
            outputs.append(output_mini_batch.transpose(1, 2).contiguous())

        Z = torch.cat(outputs, dim=1).reshape(batch_size, seq_len, self.hidden_size)
        Z = self.post_norm(Z)
        Z = F.gelu(self.wg(hidden_states)) * Z
        return self.wo(Z)


class TTTv2Block(nn.Module):
    """
    Official TTT block structure:
    pre-conv -> RMSNorm -> TTTLinear -> residual -> RMSNorm -> SwiGLU -> residual
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
        dropout_ratio: float,
        mini_batch_size: int = 16,
        conv_width: int = 4,
        ttt_base_lr: float = 1.0,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        intermediate_factor: float = 8.0 / 3.0,
        pre_conv: bool = True,
    ):
        super(TTTv2Block, self).__init__()
        del max_seq_len

        self.pre_conv = pre_conv
        self.seq_norm = RMSNorm(embedding_dim, eps=rms_norm_eps)
        self.ffn_norm = RMSNorm(embedding_dim, eps=rms_norm_eps)
        if self.pre_conv:
            self.conv = ConvModule(
                hidden_size=embedding_dim,
                conv_width=conv_width,
                eps=rms_norm_eps,
                initializer_range=initializer_range,
            )

        self.seq_modeling_block = TTTLinear(
            hidden_size=embedding_dim,
            num_heads=num_heads,
            mini_batch_size=mini_batch_size,
            conv_width=conv_width,
            rope_theta=rope_theta,
            ttt_base_lr=ttt_base_lr,
            eps=rms_norm_eps,
            initializer_range=initializer_range,
        )
        intermediate_size = max(embedding_dim, int(round(intermediate_factor * embedding_dim)))
        self.feed_forward = SwiGLUMLP(
            hidden_size=embedding_dim,
            intermediate_size=intermediate_size,
            dropout_ratio=dropout_ratio,
            initializer_range=initializer_range,
        )

    def forward(self, inputs: torch.Tensor, timestamps: Optional[np.ndarray] = None) -> torch.Tensor:
        del timestamps
        hidden_states = inputs
        if self.pre_conv:
            hidden_states = hidden_states + self.conv(hidden_states)

        hidden_states_pre_normed = self.seq_norm(hidden_states)
        seq_modeling_output = self.seq_modeling_block(hidden_states_pre_normed)
        hidden_states = hidden_states + seq_modeling_output

        feed_forward_input = self.ffn_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(feed_forward_input)
        return hidden_states


class TTTv2(nn.Module):
    """
    Graph temporal model using the official chunk/minibatch TTT-Linear block.
    """

    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        neighbor_sampler: NeighborSampler,
        num_neighbors: int,
        time_feat_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
        mini_batch_size: int = 16,
        ttt_base_lr: float = 1.0,
        conv_width: int = 4,
        pre_conv: bool = True,
        rope_theta: float = 10000.0,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        intermediate_factor: float = 8.0 / 3.0,
    ):
        super(TTTv2, self).__init__()

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
            "node": nn.Linear(in_features=self.node_feat_dim, out_features=self.embedding_dim, bias=True),
            "edge": nn.Linear(in_features=self.edge_feat_dim, out_features=self.embedding_dim, bias=True),
            "time": nn.Linear(in_features=self.time_feat_dim, out_features=self.embedding_dim, bias=True),
            "ID": nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim, bias=True),
        })

        hidden_dim = 4 * self.embedding_dim
        seq_len = self.num_neighbors + 1
        self.ttt_blocks = nn.ModuleList([
            TTTv2Block(
                embedding_dim=hidden_dim,
                num_heads=self.num_heads,
                max_seq_len=seq_len,
                dropout_ratio=self.dropout,
                mini_batch_size=max(1, int(mini_batch_size)),
                conv_width=max(1, int(conv_width)),
                ttt_base_lr=ttt_base_lr,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                initializer_range=initializer_range,
                intermediate_factor=intermediate_factor,
                pre_conv=pre_conv,
            )
            for _ in range(self.num_layers)
        ])

        self.output_layer_src = nn.Linear(in_features=hidden_dim, out_features=self.embedding_dim, bias=True)
        self.output_layer_dst = nn.Linear(in_features=2 * self.embedding_dim, out_features=self.embedding_dim, bias=True)

    def compute_src_node_temporal_embeddings(
        self,
        src_node_ids: np.ndarray,
        node_interact_times: np.ndarray,
        num_neighbors: int = 20,
    ):
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

        for ttt_block in self.ttt_blocks:
            src_node_features = ttt_block(inputs=src_node_features, timestamps=src_neighbor_times)

        return self.output_layer_src(src_node_features[:, -1, :])

    def compute_dst_node_temporal_embeddings(self, dst_node_ids: np.ndarray):
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(dst_node_ids)]
        nodes_neighbor_ID_features = self.ID_embedding(torch.tensor(dst_node_ids).to(self.device))

        dst_nodes_neighbor_node_raw_features = self.projection_layer["node"](nodes_neighbor_node_raw_features)
        dst_nodes_neighbor_ID_features = self.projection_layer["ID"](nodes_neighbor_ID_features)

        dst_node_features = torch.cat([dst_nodes_neighbor_node_raw_features, dst_nodes_neighbor_ID_features], dim=-1)
        return self.output_layer_dst(dst_node_features)

    def get_features(
        self,
        node_interact_times: np.ndarray,
        nodes_neighbor_ids: np.ndarray,
        nodes_edge_ids: np.ndarray,
        nodes_neighbor_times: np.ndarray,
        time_encoder: TimeEncoder,
    ):
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
