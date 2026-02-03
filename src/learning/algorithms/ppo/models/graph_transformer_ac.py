import torch
from torch_geometric.data import Data, Batch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import TransformerConv
import matplotlib.pyplot as plt

from learning.algorithms.ppo.models.utils import create_graph_batch


class GraphTransformerLayer(nn.Module):
    def __init__(
        self, in_dim, out_dim, heads=4, dropout=0.1, edge_dim=None, concat=True
    ):
        super(GraphTransformerLayer, self).__init__()
        self.transformer = TransformerConv(
            in_dim,
            out_dim // heads if concat else out_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            concat=concat,
            beta=True,  # Use the bias
        )
        self.layer_norm = nn.LayerNorm(out_dim if concat else out_dim)
        self.concat = concat
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=False):
        # TransformerConv with residual connection and layer norm
        if return_attention_weights:
            res, attention_weights = self.transformer(
                x, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
            out = F.gelu(res) + x  # Residual connection
            out = self.layer_norm(out)
            return out, attention_weights
        else:
            res = self.transformer(x, edge_index, edge_attr=edge_attr)
            out = F.gelu(res) + x  # Residual connection
            out = self.layer_norm(out)
            return out


class ActorCritic(nn.Module):
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
        hidden_dim=128,
        n_layers=2,
        heads=2,
        graph_type="chain",
    ):
        super(ActorCritic, self).__init__()

        self.n_agents_eval = n_agents_eval
        self.d_action = d_action
        self.device = device
        self.graph_type = graph_type

        # Learnable action std
        self.log_action_std = nn.Parameter(
            torch.ones(
                d_action * n_agents_train,
                requires_grad=True,
                device=device,
            )
            * -0.5
        )

        # Initial state embedding
        self.state_embedding = nn.Linear(d_state, hidden_dim)

        # Graph Transformer layers
        self.transformer_layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.transformer_layers.append(
                GraphTransformerLayer(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    heads=heads,
                    dropout=0.1,
                    concat=False,  # Keep dimension consistent
                )
            )

        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_action),
        )

        # Critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        # Global pooling via attention
        self.att_pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))
        )

    def get_action_and_value(self, state):
        graph_list = create_graph_batch(state, self.graph_type, self.device)
        batched_graph = Batch.from_data_list(graph_list)

        x = self.forward(batched_graph)

        graph_emb = self.att_pool(x, batched_graph.batch)
        value = self.value_head(graph_emb)

        action_mean = (
            self.actor_head(x)
            .reshape((state.shape[0], state.shape[1], self.d_action))
            .flatten(start_dim=1)
        )

        return action_mean, value

    def forward(self, batch: Batch):
        # Initial embedding
        x = self.state_embedding(batch.x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, batch.edge_index)

        return x

    def forward_evaluation(self, batch: Batch):
        # Initial embedding
        x = self.state_embedding(batch.x)

        attention_layers = []

        # Apply transformer layers and collect attention weights
        for layer in self.transformer_layers:
            x, attention_weights = layer(
                x, batch.edge_index, return_attention_weights=True
            )
            # Properly clone and detach to avoid memory corruption
            # Extract (edge_index, attn_weights) tuple
            edge_index, attn_weights = attention_weights

            # Create a deep copy of the tensors to avoid memory issues
            edge_index_copy = edge_index.clone().detach()
            attn_weights_copy = attn_weights.clone().detach()

            attention_layers.append((edge_index_copy, attn_weights_copy))

        return x, attention_layers

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            _, value = self.get_action_and_value(state)
            return value

    def get_action_dist(self, action_mean):
        action_std = torch.exp(self.log_action_std[: action_mean.shape[-1]])
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):
        action_mean, value = self.get_action_and_value(state)

        if deterministic:
            return action_mean.detach()

        dist = self.get_action_dist(action_mean)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state, action):
        action_mean, value = self.get_action_and_value(state)

        dist = self.get_action_dist(action_mean)

        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return action_logprobs, value, dist_entropy

    def get_batched_graph(self, x):
        graph_list = create_graph_batch(x, self.graph_type, self.device)
        return Batch.from_data_list(graph_list)


if __name__ == "__main__":
    from learning.plotting.utils import (
        plot_attention_heatmap,
        plot_attention_time_series,
        plot_gat_attention_as_graph,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    graph_type = "chain"
    model = ActorCritic(
        n_agents_train=8,
        n_agents_eval=8,
        d_state=24,
        d_action=2,
        device=device,
        graph_type=graph_type,
    ).to(device)

    model.eval()

    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {pytorch_total_params}")

    # Test with toy data
    x = torch.randn(1, 8, 18).to(device)  # 8 nodes, 18-d features
    graph_list = create_graph_batch(x, model.graph_type, model.device)
    batched_graph = Batch.from_data_list(graph_list)

    # Forward pass with attention collection
    x, attention_layers = model.forward_evaluation(batched_graph)

    # Plot first layer attention
    edge_index, attention_weights = attention_layers[0]

    fig = plot_gat_attention_as_graph(edge_index, attention_weights)
    plt.savefig("graph_transformer_attention_layer1.png", dpi=300, bbox_inches="tight")

    fig = plot_attention_heatmap(edge_index, attention_weights)
    plt.savefig("graph_transformer_heatmap_layer1.png", dpi=300, bbox_inches="tight")

    # Plot second layer attention
    edge_index, attention_weights = attention_layers[1]

    fig = plot_gat_attention_as_graph(edge_index, attention_weights)
    plt.savefig("graph_transformer_attention_layer2.png", dpi=300, bbox_inches="tight")

    fig = plot_attention_heatmap(edge_index, attention_weights)
    plt.savefig("graph_transformer_heatmap_layer2.png", dpi=300, bbox_inches="tight")

    # Plot attention through time
    edge_indices = []
    attention_weights = []

    # Run model over multiple timesteps
    for t in range(100):
        x_t = torch.randn(1, 8, 18).to(device)
        graph_list = create_graph_batch(x_t, model.graph_type, model.device)
        batched_graph = Batch.from_data_list(graph_list)

        # Get attention weights
        _, attention_layers = model.forward_evaluation(batched_graph)
        edge_index, attn_weight = attention_layers[0]

        edge_indices.append(edge_index)
        attention_weights.append(attn_weight)

    fig = plot_attention_time_series(edge_indices, attention_weights, top_k=5)
    plt.savefig("graph_transformer_attention_time_series.png", dpi=300)
