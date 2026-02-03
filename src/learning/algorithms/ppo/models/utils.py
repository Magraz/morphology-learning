import torch
from torch_geometric.data import Data


def create_chain_graph_batch(x_tensor, device):
    """Convert a batched tensor into a list of chain graphs with self-loops."""
    graphs = []

    for g in range(x_tensor.size(0)):
        x = x_tensor[g]  # (n_nodes, feat_dim)
        n_nodes = x.size(0)

        # Chain edges: i <-> i+1
        edges = [[i, i + 1] for i in range(n_nodes - 1)]
        edges += [[i + 1, i] for i in range(n_nodes - 1)]

        # Add self-loops (i -> i)
        edges += [[i, i] for i in range(n_nodes)]

        edge_index = (
            torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        )  # (2, E)

        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs


def create_fully_connected_graph_batch(x_tensor, device):
    """Convert a batched tensor into a list of fully connected graphs using vectorized operations."""
    graphs = []

    for g in range(x_tensor.size(0)):
        x = x_tensor[g]  # (n_nodes, feat_dim)
        n_nodes = x.size(0)

        # Create all pairs of indices for fully connected graph using meshgrid
        rows, cols = torch.meshgrid(
            torch.arange(n_nodes, device=device),
            torch.arange(n_nodes, device=device),
            indexing="ij",
        )

        # Stack to get edge_index format (2, n_nodes²)
        edge_index = torch.stack([rows.flatten(), cols.flatten()], dim=0)

        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs


def create_graph_batch(input: torch.Tensor, type: str, device: str):
    match (type):
        case "full":
            return create_fully_connected_graph_batch(input, device)
        case "chain":
            return create_chain_graph_batch(input, device)
