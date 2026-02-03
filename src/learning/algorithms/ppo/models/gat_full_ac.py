import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import AttentionalAggregation
import matplotlib.pyplot as plt

from learning.algorithms.ppo.models.utils import create_graph_batch


class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
        hidden_dim=128,
        graph_type="full",
    ):
        super(ActorCritic, self).__init__()

        self.n_agents_eval = n_agents_eval
        self.d_action = d_action
        self.device = device
        self.graph_type = graph_type

        self.log_action_std = nn.Parameter(
            torch.ones(
                d_action * n_agents_train,
                requires_grad=True,
                device=device,
            )
            * -0.5
        )

        self.gat1 = GATv2Conv(d_state, hidden_dim, heads=2, concat=True)
        self.gat2 = GATv2Conv(hidden_dim * 2, hidden_dim, heads=1, concat=False)

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
        x = self.gat1(batch.x, batch.edge_index)
        x = F.gelu(x)
        # Normalization is important for GAT training stability
        x = F.layer_norm(x, x.shape[1:])
        x = self.gat2(x, batch.edge_index)
        x = F.gelu(x)

        return x

    def forward_evaluation(self, batch: Batch):
        x, att1 = self.gat1(batch.x, batch.edge_index, return_attention_weights=True)
        x = F.gelu(x)
        # Normalization is important for GAT training stability
        x = F.layer_norm(x, x.shape[1:])
        x, att2 = self.gat2(x, batch.edge_index, return_attention_weights=True)
        x = F.gelu(x)

        # Move everything to CPU and convert to numpy to completely break connections
        # For first attention layer
        edge_index1, attn_weights1 = att1
        edge_index1_safe = edge_index1.clone().detach().cpu()
        attn_weights1_safe = attn_weights1.clone().detach().cpu()

        # For second attention layer
        edge_index2, attn_weights2 = att2
        edge_index2_safe = edge_index2.clone().detach().cpu()
        attn_weights2_safe = attn_weights2.clone().detach().cpu()

        # Create safe copies that won't affect the computation graph
        att_layers = [
            (edge_index1_safe, attn_weights1_safe),
            (edge_index2_safe, attn_weights2_safe),
        ]

        return x, att_layers

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
        graph_list = create_graph_batch(x, self.graph_type)
        return Batch.from_data_list(graph_list)


if __name__ == "__main__":
    from learning.plotting.utils import (
        plot_attention_heatmap,
        plot_attention_time_series,
        plot_gat_attention_as_graph,
        plot_diagonal_attention_timeline,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    graph_type = "chain"

    model = ActorCritic(
        n_agents_train=8,
        n_agents_eval=8,
        d_state=18,
        d_action=2,
        device=device,
        graph_type=graph_type,
    ).to(device)

    model.eval()

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # toy graph ----------------------------------------------------
    x = torch.randn(1, 8, 18).to(device)  # 8 nodes, 18-d features
    graph_list = create_graph_batch(x, model.graph_type, model.device)

    batched_graph = Batch.from_data_list(graph_list)

    x, attention_layers = model.forward_evaluation(batched_graph)

    # Plot first layer attention (ei is edge_index, alpha is attention weights)
    edge_index, attention_weights = attention_layers[0]

    fig = plot_gat_attention_as_graph(edge_index, attention_weights)
    plt.savefig("gat_attention_layer1.png", dpi=300, bbox_inches="tight")

    fig = plot_attention_heatmap(edge_index, attention_weights)
    plt.savefig("attention_heatmap_layer1.png", dpi=300, bbox_inches="tight")

    # Plot second layer attention
    edge_index, attention_weights = attention_layers[1]

    fig = plot_gat_attention_as_graph(edge_index, attention_weights)
    plt.savefig("gat_attention_layer2.png", dpi=300, bbox_inches="tight")

    fig = plot_attention_heatmap(edge_index, attention_weights)
    plt.savefig("attention_heatmap_layer2.png", dpi=300, bbox_inches="tight")

    # Plot through time
    edge_indices = []
    attention_weights = []

    # Run model over multiple timesteps
    for t in range(100):  # For example, 10 timesteps
        # Your input at this timestep
        x_t = torch.randn(1, 8, 18).to(device)

        # Forward pass
        graph_list = create_graph_batch(x_t, chain_type)
        batched_graph = Batch.from_data_list(graph_list)

        # Get attention weights
        _, attention_layers = model.forward_evaluation(batched_graph)

        # Store edge indices and weights from first layer
        edge_index, attn_weight = attention_layers[0]

        edge_indices.append(edge_index)
        attention_weights.append(attn_weight)

    fig = plot_attention_time_series(edge_indices, attention_weights, top_k=5)
    plt.savefig("attention_time_series.png", dpi=300)

    fig = plot_3d_attention_surface(edge_indices, attention_weights)
    plt.savefig("attention_3D.png", dpi=300)

    ffig = plot_diagonal_attention_timeline(
        edge_indices, attention_weights, num_samples=5
    )
    plt.savefig("attention_timeline.png", dpi=300)

    # For the standard 3D scatter plot
    fig = plot_3d_attention_scatter(edge_indices, attention_weights, sample_rate=5)
    plt.savefig("attention_3d_scatter.png", dpi=300, bbox_inches="tight")

    # For the 3D volume visualization
    fig_volume = plot_3d_attention_volume(
        edge_indices, attention_weights, sample_rate=5
    )
    plt.savefig("attention_3d_volume.png", dpi=300, bbox_inches="tight")
