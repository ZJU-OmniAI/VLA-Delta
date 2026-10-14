"""Dense flow prediction from only the trailing Difference Query states."""

import torch
from torch import nn
from torch.nn import functional as F


class FlowDecoderLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden))
        self.output_norm = nn.LayerNorm(hidden)

    def forward(self, grid, queries):
        grid = self.norm(grid + self.attention(grid, queries, queries, need_weights=False)[0])
        return self.output_norm(grid + self.mlp(grid))


class DenseRegressionFlowHead(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config.validate()
        hidden = config.flow_head_hidden_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.grid = nn.Parameter(torch.empty(1, config.flow_grid_size ** 2, hidden))
        nn.init.normal_(self.grid, std=0.02)
        self.layers = nn.ModuleList([
            FlowDecoderLayer(hidden, config.flow_head_num_heads)
            for _ in range(config.flow_head_num_layers)
        ])
        self.refinement = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1),
                                        nn.GroupNorm(8, hidden), nn.GELU())
        self.output = nn.Conv2d(hidden, 2, 3, padding=1)
        nn.init.normal_(self.output.weight, std=1e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, query_states):
        if query_states.ndim != 3 or query_states.shape[1] < self.config.num_flow_queries:
            raise ValueError("flow head requires [B,Nq,H] with Nq >= num_flow_queries")
        queries = query_states[:, -self.config.num_flow_queries:, :]
        # A single differentiable boundary cast supports mixed backbone/head dtypes
        # without autocast. Parameters retain their construction or explicit .to dtype.
        queries = queries.to(dtype=self.input_projection.weight.dtype)
        queries = self.input_projection(self.input_norm(queries))
        grid = self.grid.expand(queries.shape[0], -1, -1)
        for layer in self.layers:
            grid = layer(grid, queries)
        grid = grid.transpose(1, 2).reshape(queries.shape[0], -1, self.config.flow_grid_size, self.config.flow_grid_size)
        grid = F.interpolate(grid, size=(56, 56), mode="bilinear", align_corners=False)
        return self.output(self.refinement(grid))

    def parameter_counts(self):
        return {"total": sum(p.numel() for p in self.parameters()),
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "submodules": {name: sum(p.numel() for p in module.parameters())
                               for name, module in self.named_children()},
                "grid": self.grid.numel()}


OPTICAL_FLOW_AUX_REGISTRY = {"none": None, "dense_regression_v1": DenseRegressionFlowHead}


def build_optical_flow_head(input_dim, config):
    config.validate()
    factory = OPTICAL_FLOW_AUX_REGISTRY[config.optical_flow_aux_type]
    if factory is None:
        return None
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.flow_init_seed)
        return factory(input_dim, config)
