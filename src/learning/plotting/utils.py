import matplotlib.pyplot as plt

import numpy as np

import networkx as nx

from matplotlib.patches import Rectangle
from torch_geometric.data import Batch, Data

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

import torch

# GAT plots


def plot_diagonal_attention_timeline(
    edge_indices,
    attention_weights_over_time,
    figsize=(12, 10),
    cmap="viridis",
    num_samples=5,
):
    """
    Creates a diagonal overlay visualization of attention heatmaps over time

    Args:
        edge_indices: List of edge_index tensors for each timestep
        attention_weights_over_time: List of attention weight tensors
        figsize: Size of the figure
        cmap: Colormap to use
        num_samples: Number of timesteps to sample
    """
    # Sample timesteps if needed
    total_timesteps = len(edge_indices)
    if total_timesteps > num_samples:
        timestep_indices = np.linspace(0, total_timesteps - 1, num_samples, dtype=int)
    else:
        timestep_indices = np.arange(total_timesteps)

    num_timesteps = len(timestep_indices)

    # Get maximum number of nodes
    max_nodes = max([edge_indices[i].max().item() + 1 for i in timestep_indices])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate alpha values (increasing for newer timesteps)
    alphas = np.linspace(0.3, 0.9, num_timesteps)

    # Calculate overlap percentage
    overlap_percent = 0.7

    # Size of each heatmap (adjusted to fit diagonally)
    heatmap_width = 0.8 / (1 + (num_timesteps - 1) * (1 - overlap_percent))
    heatmap_height = 0.8 / (1 + (num_timesteps - 1) * (1 - overlap_percent))

    # Plot each timestep with increasing alpha and diagonal shift
    for idx, t in enumerate(timestep_indices):
        # Calculate position for this timestep's heatmap
        pos_x = 0.1 + idx * heatmap_width * (1 - overlap_percent)
        pos_y = 0.9 - heatmap_height - idx * heatmap_height * (1 - overlap_percent)

        # Create a new axes for this timestep
        ax_t = fig.add_axes([pos_x, pos_y, heatmap_width, heatmap_height])

        # Get data for this timestep
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        # Average over heads if multiple
        if weights.shape[1] > 1:
            edge_weights = weights.mean(axis=1)
        else:
            edge_weights = weights.squeeze()

        # Create attention matrix
        attention_matrix = np.zeros((max_nodes, max_nodes))
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            if src < max_nodes and dst < max_nodes:  # Safety check
                attention_matrix[src, dst] = edge_weights[i]

        # Plot the heatmap with appropriate alpha
        im = ax_t.imshow(attention_matrix, cmap=cmap, alpha=alphas[idx], vmin=0, vmax=1)

        # Add a timestep label
        ax_t.set_title(f"t={t+1}", fontsize=10)

        # Only show axis labels for the last timestep
        if idx < num_timesteps - 1:
            ax_t.set_xticks([])
            ax_t.set_yticks([])
        else:
            ax_t.set_xlabel("Target Salp", fontsize=8)
            ax_t.set_ylabel("Source Salp", fontsize=8)

            # Add proper tick labels
            ax_t.set_xticks(np.arange(max_nodes))
            ax_t.set_yticks(np.arange(max_nodes))
            ax_t.set_xticklabels(range(max_nodes), fontsize=8)
            ax_t.set_yticklabels(range(max_nodes), fontsize=8)

    # Add a colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Weight")

    # Add a legend showing alpha progression
    legend_elements = []
    for idx, t in enumerate(timestep_indices):
        legend_elements.append(
            Rectangle(
                (0, 0),
                1,
                1,
                fc=plt.cm.get_cmap(cmap)(0.7),
                alpha=alphas[idx],
                label=f"Timestep {t+1}",
            )
        )

    # Add legend in an empty part of the figure
    legend_ax = fig.add_axes([0.05, 0.05, 0.2, 0.1])
    legend_ax.axis("off")
    legend_ax.legend(handles=legend_elements, loc="center")

    # Turn off the main axis
    ax.axis("off")

    # Set figure title
    fig.suptitle(
        "Attention Weights Evolution\n(Newer timesteps overlay with higher opacity)",
        fontsize=14,
    )

    return fig


def plot_gat_attention_as_graph(edge_index, attention_weights, figsize=(12, 10)):
    """
    Plot a graph with edges colored according to attention weights

    Args:
        edge_index: Tensor of shape [2, num_edges] containing edge connections
        attention_weights: Tensor of shape [num_edges, num_heads] containing attention scores
        figsize: Size of the figure
    """
    # Convert to numpy for easier handling
    edges = edge_index.cpu().numpy()
    weights = attention_weights.detach().cpu().numpy()

    # If there are multiple attention heads, average them
    if weights.shape[1] > 1:
        edge_weights = weights.mean(axis=1)
        print(f"Averaging {weights.shape[1]} attention heads")
    else:
        edge_weights = weights.squeeze()

    # Create a directed graph
    G = nx.DiGraph()

    # Get number of nodes from edge indices
    num_nodes = max(edges.max() + 1, 8)  # At least 8 nodes for your example

    # Add all nodes (including isolated ones)
    G.add_nodes_from(range(num_nodes))

    # Add edges with weights
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        G.add_edge(src.item(), dst.item(), weight=edge_weights[i])

    # Set up the plot - FIXED: Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=figsize)

    # Use a layout that spreads nodes nicely
    if num_nodes <= 8:  # For small graphs
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Normalize edge weights for coloring
    edge_weights_list = [G[u][v]["weight"] for u, v in G.edges()]
    min_weight = min(edge_weights_list) if edge_weights_list else 0
    max_weight = max(edge_weights_list) if edge_weights_list else 1
    normalized_weights = [
        (w - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 0.5
        for w in edge_weights_list
    ]

    # Draw the nodes - FIXED: Pass ax parameter
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue", ax=ax)

    # Draw the edges with a colormap based on weight - FIXED: Pass ax parameter
    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle="-|>",
        arrowsize=20,
        edge_color=normalized_weights,
        edge_cmap=plt.cm.Blues,
        width=4,
        ax=ax,
    )

    # Add a colorbar - FIXED: Pass ax parameter to colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_weight, vmax=max_weight)
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Attention Weight")

    # Add labels - FIXED: Pass ax parameter
    nx.draw_networkx_labels(G, pos, font_size=14, ax=ax)

    ax.set_title("Graph Attention Visualization", fontsize=16)
    ax.axis("off")
    plt.tight_layout()

    # Return the figure for further customization
    return fig


def plot_attention_heatmap(
    model_name, edge_index, attention_weights, figsize=(10, 8), cmap="PuBu"
):
    """
    Plot attention weights as a matrix heatmap using seaborn

    Args:
        model_name: Name of the model for title
        edge_index: Tensor of shape [2, num_edges] containing edge connections
        attention_weights: Tensor of shape [num_edges, num_heads] containing attention scores
        figsize: Size of the figure
        cmap: Colormap to use
    """
    # Convert to numpy for easier handling
    edges = edge_index.cpu().numpy()
    weights = attention_weights.detach().cpu().numpy()

    # If there are multiple attention heads, average them
    if weights.shape[1] > 1:
        edge_weights = weights.mean(axis=1)
        print(f"Averaging {weights.shape[1]} attention heads")
    else:
        edge_weights = weights.squeeze()

    # Get number of nodes
    num_nodes = max(edges.max() + 1, 4)

    # Create an empty attention matrix
    attention_matrix = np.zeros((num_nodes, num_nodes))

    # Fill the matrix with attention weights
    for i in range(edges.shape[1]):
        src, dst = edges[0, i], edges[1, i]
        attention_matrix[src, dst] = edge_weights[i]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap using seaborn
    sns.heatmap(
        attention_matrix,
        annot=True,  # Show values in cells
        fmt=".2f",  # Format for values
        cmap=cmap,  # Use the same colormap as transformer attention
        linewidths=0.5,  # Add grid lines between cells
        cbar=True,  # Show colorbar
        ax=ax,
    )

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set y-tick labels to horizontal
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Set labels and title
    ax.set_title(f"Attention Scores Matrix - {model_name}")
    ax.set_xlabel("Target Salp")
    ax.set_ylabel("Source Salp")

    # Set custom tick labels
    # Note: seaborn's heatmap already centers ticks in cells
    ax.set_xticklabels([f"Salp {i}" for i in range(num_nodes)])
    ax.set_yticklabels([f"Salp {i}" for i in range(num_nodes)])

    plt.tight_layout()
    return fig


def plot_attention_time_series(edge_indices, attention_weights_over_time, top_k=5):
    """
    Plot the attention weights over time for the top-k most important edges

    Args:
        edge_indices: List of edge_index tensors
        attention_weights_over_time: List of attention weight tensors
        top_k: Number of top edges to track
    """
    num_timesteps = len(edge_indices)

    # Find edges that appear in all timesteps
    # For simplicity, we'll assume the graph structure stays constant
    edges = edge_indices[0].cpu().numpy()
    edge_pairs = [(edges[0, i], edges[1, i]) for i in range(edges.shape[1])]

    # Calculate average attention over time for each edge
    edge_avg_attention = np.zeros(len(edge_pairs))
    for t in range(num_timesteps):
        weights = attention_weights_over_time[t].detach().cpu().numpy()
        if weights.shape[1] > 1:  # If multiple heads
            weights = weights.mean(axis=1)
        else:
            weights = weights.squeeze()

        edge_avg_attention += weights

    edge_avg_attention /= num_timesteps

    # Get indices of top-k edges by average attention
    top_indices = np.argsort(edge_avg_attention)[-top_k:]

    # Prepare time series data
    time_steps = np.arange(1, num_timesteps + 1)
    attention_series = np.zeros((top_k, num_timesteps))

    for t in range(num_timesteps):
        weights = attention_weights_over_time[t].detach().cpu().numpy()
        if weights.shape[1] > 1:
            weights = weights.mean(axis=1)
        else:
            weights = weights.squeeze()

        for i, idx in enumerate(top_indices):
            attention_series[i, t] = weights[idx]

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, idx in enumerate(top_indices):
        src, dst = edge_pairs[idx]
        ax.plot(
            time_steps,
            attention_series[i],
            linewidth=2,
            label=f"Edge {src}→{dst}",
        )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title("Attention Evolution for Top Edges", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def plot_node_attention_trends(
    experiment,
    edge_indices,
    attention_weights_over_time,
    source_node_idx=0,
    figsize=(12, 8),
):
    """Plot how one specific node attends to all other nodes over time"""
    num_timesteps = len(edge_indices)

    # Determine the maximum node index
    max_node_idx = 0
    for t in range(num_timesteps):
        max_node_idx = max(max_node_idx, edge_indices[t].max().item())

    # Create dictionary to track attention from source to each target
    target_attention = {}

    # Process each timestep
    for t in range(num_timesteps):
        edges = edge_indices[t].cpu().numpy()
        weights = attention_weights_over_time[t].detach().cpu().numpy()

        # Average over heads if multiple
        if len(weights.shape) > 1 and weights.shape[1] > 1:
            weights = weights.mean(axis=1)
        else:
            weights = weights.squeeze()

        # Find all edges where source_node_idx is the source
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            if src == source_node_idx:
                if dst not in target_attention:
                    target_attention[dst] = [0] * num_timesteps
                target_attention[dst][t] = weights[i]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create improved color generation for more distinct colors
    num_targets = len(target_attention)

    # Use multiple colormaps combined for greater variety
    if num_targets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
    elif num_targets <= 20:
        # Combine two colormaps for more colors
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.Set1(np.linspace(0, 1, 10))
        colors = np.vstack([colors1, colors2])
    else:
        # For many targets, use HSV color space for more distinct colors
        cmap = plt.cm.hsv
        colors = cmap(
            np.linspace(0, 0.9, num_targets)
        )  # Stop at 0.9 to avoid red-red cycle

    # Plot each target's attention over time
    time_steps = np.arange(1, num_timesteps + 1)

    for i, (target, values) in enumerate(sorted(target_attention.items())):
        # Use dashed line for self-attention
        linestyle = "--" if target == source_node_idx else "-"
        linewidth = 2.5 if target == source_node_idx else 1.8
        color_idx = i % len(colors)

        label = f"Salp {source_node_idx} → Salp {target}"
        if target == source_node_idx:
            label += " (self)"

        ax.plot(
            time_steps,
            values,
            linewidth=linewidth,
            linestyle=linestyle,
            marker="o" if target == source_node_idx else None,
            markersize=4,
            color=colors[color_idx],
            label=label,
        )

    # Add labels and title
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)
    ax.set_title(
        f"{experiment}: Salp {source_node_idx} Attention to All Nodes Over Time",
        fontsize=14,
    )

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


# Transformer plots


def plot_transformer_attention(
    model_name, attn_matrix, layer_name="", head_idx=0, figsize=(10, 8)
):
    """
    Plot a single attention head as a heatmap

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        head_idx: Which attention head to visualize
        figsize: Size of the figure
    """
    # Extract the first batch item and specified head
    attention = attn_matrix[0, head_idx].cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        attention,
        annot=True,
        fmt=".2f",
        cmap="PuBu",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Set y-tick labels to horizontal
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Set labels and title
    ax.set_xlabel("Target Salp")
    ax.set_ylabel("Source Salp")
    ax.set_title(f"Attention Scores Matrix - {model_name}")

    # Custom tick labels - FIXED: explicitly set tick positions first
    n_agents = attention.shape[0]

    # Now set the labels for those positions
    ax.set_xticklabels([f"Salp {i+1}" for i in range(n_agents)])
    ax.set_yticklabels([f"Salp {i+1}" for i in range(n_agents)])

    plt.tight_layout()
    return fig


def plot_all_attention_heads(attn_matrix, layer_name="", figsize=(15, 12)):
    """
    Plot all attention heads in a grid

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        figsize: Size of the figure
    """
    batch_size, n_heads, seq_len, _ = attn_matrix.shape

    # Create grid layout based on number of heads
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for h in range(n_heads):
        if h < len(axes):
            attention = attn_matrix[0, h].cpu().numpy()

            sns.heatmap(
                attention,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                linewidths=0.5,
                cbar=True if h == 0 else False,
                ax=axes[h],
            )

            axes[h].set_title(f"Head {h+1}")

            if h % n_cols == 0:  # Leftmost plots
                axes[h].set_ylabel("Source Agent")
            else:
                axes[h].set_ylabel("")

            if h >= n_heads - n_cols:  # Bottom row
                axes[h].set_xlabel("Target Agent")
            else:
                axes[h].set_xlabel("")

            # Custom tick labels
            axes[h].set_xticklabels([f"{i+1}" for i in range(seq_len)], rotation=45)
            axes[h].set_yticklabels([f"{i+1}" for i in range(seq_len)])

    # Hide unused subplots
    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"{layer_name} - All Attention Heads", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_averaged_attention(attn_matrix, layer_name="", figsize=(10, 8)):
    """
    Plot attention averaged across all heads

    Args:
        attn_matrix: Attention matrix with shape [B, H, L, L]
        layer_name: Name of the layer (for title)
        figsize: Size of the figure
    """
    # Average over heads dimension
    avg_attention = attn_matrix[0].mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        avg_attention,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        linewidths=0.5,
        cbar=True,
        ax=ax,
    )

    # Set labels and title
    ax.set_xlabel("Target Agent")
    ax.set_ylabel("Source Agent")
    ax.set_title(f"{layer_name} - Average Attention Across All Heads")

    # Custom tick labels
    n_agents = avg_attention.shape[0]
    ax.set_xticklabels([f"{i+1}" for i in range(n_agents)])
    ax.set_yticklabels([f"{i+1}" for i in range(n_agents)])

    plt.tight_layout()
    return fig


def plot_attention_across_layers(attn_scores_dict, figsize=(15, 10)):
    """
    Plot average attention for each layer side by side

    Args:
        attn_scores_dict: Dictionary of attention matrices {layer_name: attn_matrix}
        figsize: Size of the figure
    """
    n_layers = len(attn_scores_dict)

    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]

    for i, (layer_name, attn_matrix) in enumerate(attn_scores_dict.items()):
        # Average over heads dimension
        avg_attention = attn_matrix[0].mean(dim=0).cpu().numpy()

        # Create heatmap
        sns.heatmap(
            avg_attention,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            linewidths=0.5,
            cbar=True if i == n_layers - 1 else False,
            ax=axes[i],
        )

        axes[i].set_title(f"{layer_name}")
        axes[i].set_xlabel("Target Agent")

        if i == 0:  # Only first plot shows y-label
            axes[i].set_ylabel("Source Agent")

        # Custom tick labels
        n_agents = avg_attention.shape[0]
        axes[i].set_xticklabels([f"{i+1}" for i in range(n_agents)])
        axes[i].set_yticklabels([f"{i+1}" for i in range(n_agents)])

    fig.suptitle("Attention Patterns Across Layers", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def plot_attention_over_time_grid(
    attention_over_time, attn_type="Enc_L0", head_idx=0, num_samples=4, figsize=(20, 6)
):
    """
    Plot attention at selected timesteps in a grid

    Args:
        attention_over_time: Dict with lists of attention weights per timestep
        attn_type: Type of attention to visualize
        head_idx: Which attention head to visualize
        num_samples: Number of timesteps to sample and display
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    attention_matrices = attention_over_time[attn_type]
    total_timesteps = len(attention_matrices)

    # Sample evenly spaced timesteps
    if total_timesteps > num_samples:
        sample_indices = np.linspace(0, total_timesteps - 1, num_samples, dtype=int)
    else:
        sample_indices = np.arange(total_timesteps)

    # Create figure with subplots
    fig, axes = plt.subplots(1, len(sample_indices), figsize=figsize)
    if len(sample_indices) == 1:
        axes = [axes]

    # Plot each sampled timestep
    for i, t_idx in enumerate(sample_indices):
        attn_matrix = attention_matrices[t_idx][0, head_idx].cpu().numpy()

        sns.heatmap(
            attn_matrix,
            annot=True,
            fmt=".2f",
            cmap="PuBu",
            vmin=0,
            vmax=1,
            cbar=(i == len(sample_indices) - 1),  # Only add colorbar on last plot
            ax=axes[i],
        )

        axes[i].set_title(f"Timestep {t_idx+1}")

        if i == 0:  # Only add y-label to leftmost plot
            axes[i].set_ylabel("Source Position")
        else:
            axes[i].set_ylabel("")

        axes[i].set_xlabel("Target Position")

        # Add tick labels
        n_positions = attn_matrix.shape[0]
        axes[i].set_xticklabels([f"{j+1}" for j in range(n_positions)], rotation=45)
        axes[i].set_yticklabels([f"{j+1}" for j in range(n_positions)])

    # Add overall title
    type_label = {
        "Enc_L0": "Encoder Self-Attention",
        "Dec_L0": "Decoder Self-Attention",
        "Cross_L0": "Cross-Attention",
    }
    fig.suptitle(
        f"{type_label.get(attn_type, attn_type)} - Head {head_idx+1} Evolution",
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_key_attention_trends(
    attention_over_time, attn_type="Enc_L0", head_idx=0, top_k=5, figsize=(12, 8)
):
    """
    Plot the top-k attention connections over time as a line chart

    Args:
        attention_over_time: Dict with lists of attention weights per timestep
        attn_type: Type of attention to visualize ('Enc_L0', 'Dec_L0', or 'Cross_L0')
        head_idx: Which attention head to visualize
        top_k: Number of top connections to track
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the attention matrices for the specified type
    attention_matrices = attention_over_time[attn_type]

    if not attention_matrices:
        print(f"No attention data found for {attn_type}")
        return plt.figure(figsize=(6, 4))

    num_timesteps = len(attention_matrices)

    # Get matrix dimensions from first timestep
    batch_size, num_heads, seq_len, _ = attention_matrices[0].shape

    # Calculate average attention over time for each source-target pair
    # to determine which ones are most important
    avg_attention = torch.zeros((seq_len, seq_len))

    for t in range(num_timesteps):
        avg_attention += attention_matrices[t][0, head_idx].cpu()

    avg_attention /= num_timesteps

    # Find top-k connections
    flat_indices = torch.topk(avg_attention.view(-1), top_k).indices
    top_src = flat_indices // seq_len
    top_tgt = flat_indices % seq_len

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Line colors
    colors = plt.cm.viridis(np.linspace(0, 1, top_k))

    # Prepare time series data
    time_steps = np.arange(1, num_timesteps + 1)

    # Plot each top connection
    for i in range(top_k):
        src, tgt = top_src[i].item(), top_tgt[i].item()
        values = []

        for t in range(num_timesteps):
            values.append(attention_matrices[t][0, head_idx, src, tgt].item())

        ax.plot(
            time_steps,
            values,
            markersize=4,
            linewidth=2,
            color=colors[i],
            label=f"Src {src+1} → Tgt {tgt+1}",
        )

    # Add labels and title
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Attention Weight", fontsize=12)

    type_label = {
        "Enc_L0": "Encoder Self-Attention",
        "Dec_L0": "Decoder Self-Attention",
        "Cross_L0": "Cross-Attention",
    }

    ax.set_title(
        f"{type_label.get(attn_type, attn_type)} - Head {head_idx+1}\nTop {top_k} Attention Connections",
        fontsize=14,
    )

    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    # Format y-axis ticks as percentages
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Add horizontal lines at key thresholds
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=0.75, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    return fig


def plot_token_attention_trends(
    experiment,
    n_agents,
    attention_over_time,
    attn_type="Enc_L0",
    src_idx=0,
    head_idx=0,
    figsize=(12, 8),
    min_threshold=0.05,  # Add threshold parameter
    filter_by="max",  # Filter by max or avg attention
):
    """
    Plot how one specific token attends to all other tokens over time.
    """
    if attn_type not in attention_over_time:
        print(f"Attention type {attn_type} not found in data")
        return None

    # Get data
    timesteps = attention_over_time[attn_type]
    num_timesteps = len(timesteps)

    # Check first timestep's shape to validate indexing
    first_tensor = timesteps[0]
    if len(first_tensor.shape) != 4:
        print(
            f"Unexpected tensor shape: {first_tensor.shape}. Expected (batch, heads, seq, seq)"
        )
        return None

    # Check if we have enough heads
    if first_tensor.shape[1] <= head_idx:
        print(
            f"Warning: Head index {head_idx} out of bounds. Model has {first_tensor.shape[1]} heads"
        )
        return None

    # Create dictionary to track token's attention to each target
    token_attention = {}

    # Extract attention patterns for the source token over time
    for t in range(num_timesteps):
        try:
            # Get attention matrix for this timestep and head - FIXED INDEXING HERE
            attn_matrix = timesteps[t][
                0, head_idx
            ].cpu()  # Add batch dimension index [0]

            # Extract row corresponding to source token (its attention to all tokens)
            if src_idx < attn_matrix.shape[0]:
                src_attention = attn_matrix[src_idx, :]

                # Store attention to each target
                for tgt_idx in range(src_attention.shape[0]):
                    if tgt_idx not in token_attention:
                        token_attention[tgt_idx] = [0] * num_timesteps

                    # Extract value safely
                    if isinstance(src_attention[tgt_idx], torch.Tensor):
                        if src_attention[tgt_idx].numel() == 1:
                            token_attention[tgt_idx][t] = src_attention[tgt_idx].item()
                        else:
                            token_attention[tgt_idx][t] = (
                                src_attention[tgt_idx].mean().item()
                            )
                    else:
                        token_attention[tgt_idx][t] = src_attention[tgt_idx]
            else:
                print(f"Warning: Source index {src_idx} out of bounds for {attn_type}")
        except Exception as e:
            print(f"Error processing {attn_type}, head {head_idx}, timestep {t}: {e}")
            continue

    # Skip if we couldn't extract any attention patterns
    if not token_attention:
        print(
            f"No valid attention patterns found for {attn_type}, head {head_idx}, source {src_idx}"
        )
        return None

    # Filter targets with consistently low attention
    filtered_token_attention = {}
    for target, values in token_attention.items():
        if filter_by.lower() == "max":
            metric = max(values)
        else:  # 'avg'
            metric = sum(values) / len(values)

        # Keep this target if it meets the threshold or is self-attention
        if metric >= min_threshold or target == src_idx:
            filtered_token_attention[target] = values

    # Print filtering statistics
    filtered_out = len(token_attention) - len(filtered_token_attention)
    if filtered_out > 0:
        print(
            f"Filtered out {filtered_out} targets with {filter_by} attention below {min_threshold}"
        )

    # Use filtered data for plotting
    token_attention = filtered_token_attention

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Improved color generation
    num_targets = len(token_attention)

    # Create colors using different approaches based on number of targets
    if num_targets <= 10:
        # Use tab10 but with high-contrast ordering
        base_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        color_order = [i for pair in zip(range(5), range(5, 10)) for i in pair]
        colors = base_colors[color_order[:num_targets]]
    elif num_targets <= 20:
        # For more targets, combine colormaps in alternating pattern
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.Set2(np.linspace(0, 1, 8))
        colors3 = plt.cm.Dark2(np.linspace(0, 1, 8))
        colors = []
        for i in range(max(10, 8, 8)):
            if i < 10:
                colors.append(colors1[i])
            if i < 8:
                colors.append(colors2[i])
            if i < 8:
                colors.append(colors3[i])
        colors = np.array(colors[:num_targets])
    else:
        # For many targets, use HSV with alternating brightness/saturation
        cmap = plt.cm.hsv
        primary_colors = cmap(np.linspace(0, 0.8, num_targets // 2))
        secondary_colors = cmap(np.linspace(0.4, 1.0, num_targets - num_targets // 2))
        colors = []
        for i in range(max(len(primary_colors), len(secondary_colors))):
            if i < len(primary_colors):
                colors.append(primary_colors[i])
            if i < len(secondary_colors):
                colors.append(secondary_colors[i])
        colors = np.array(colors[:num_targets])

    # Plot attention over time for each target token
    time_steps = np.arange(1, num_timesteps + 1)

    # Get sorted target indices
    sorted_targets = sorted(token_attention.keys())

    # Create a mapping to specifically alternate colors
    color_mapping = {}
    for i, target in enumerate(sorted_targets):
        # Map indices to maximize visual difference
        if num_targets <= 10:
            color_mapping[target] = i
        else:
            color_mapping[target] = (i * 7) % num_targets

    # Plot using the color mapping
    for target, values in token_attention.items():
        # Special formatting for self-attention
        linestyle = "--" if target == src_idx else "-"
        linewidth = 2.5 if target == src_idx else 1.8
        marker = "o" if target == src_idx else None
        markersize = 4 if target == src_idx else 0

        # Use the color mapping to get the color index
        color_idx = color_mapping[target]

        label = f"Salp {src_idx} → Salp {target}"
        if target == src_idx:
            label += " (self)"

        ax.plot(
            time_steps,
            values,
            linewidth=linewidth,
            linestyle=linestyle,
            marker=marker,
            markersize=markersize,
            color=colors[color_idx % len(colors)],
            label=label,
        )

    # Add labels and title
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Attention Weight")
    # ax.set_title(
    #     f"{experiment}: {attn_type} Head {head_idx}, Token {src_idx} Attention",
    #     fontsize=14,
    # )

    ax.set_title(f"Salp #{src_idx} in {n_agents}-Unit Chain Attention Over Time\n")

    # Add grid and legend
    ax.grid(True, alpha=0.3)

    # Adjust legend for large number of targets
    if num_targets > 15:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    filename = f"{experiment}_{attn_type}_head{head_idx}_token{src_idx}_attention_{n_agents}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    return fig


# GCN


def visualize_gcn_relationships_over_time(model, state_sequence, figsize=(14, 10)):
    """
    Visualize how GCN node relationships evolve through time by tracking feature similarity.

    Args:
        model: Your GCN model
        state_sequence: List of state tensors representing timesteps [t, batch, n_nodes, features]
        figsize: Figure size (width, height)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import networkx as nx
    import seaborn as sns
    from matplotlib.animation import FuncAnimation

    num_timesteps = len(state_sequence)
    n_nodes = state_sequence[0].shape[1]

    # Storage for embeddings and similarities over time
    all_embeddings = []
    all_similarities = []

    # Process embeddings for each timestep
    with torch.no_grad():
        for t in range(num_timesteps):
            # Get graph for this timestep
            graph_list = model.create_chain_graph_batch(state_sequence[t])
            batched_graph = Batch.from_data_list(graph_list)

            # Get node embeddings
            embeddings = model.forward(batched_graph).cpu().numpy()
            all_embeddings.append(embeddings)

            # Calculate similarity matrix for the first batch item
            graph_embeddings = embeddings[:n_nodes]  # First graph only

            sim_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    sim_matrix[i, j] = np.dot(
                        graph_embeddings[i], graph_embeddings[j]
                    ) / (
                        np.linalg.norm(graph_embeddings[i])
                        * np.linalg.norm(graph_embeddings[j])
                    )

            all_similarities.append(sim_matrix)

    # Create figure with two subplots - graph on left, timeline on right
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Create graph for visualization
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)

    # Add edges between all nodes (will visualize strength via width/color)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            G.add_edge(i, j)

    # Use circular layout for better visualization
    pos = nx.circular_layout(G)

    # Select a few interesting node pairs to track over time
    # 1. Find pairs with highest variance in similarity
    pair_variance = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            similarities = [sim[i, j] for sim in all_similarities]
            pair_variance.append(((i, j), np.var(similarities)))

    # Sort by variance and take top 5
    top_pairs = sorted(pair_variance, key=lambda x: x[1], reverse=True)[:5]

    # Setup timeline plot
    ax2.set_xlim(0, num_timesteps)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_xlabel("Timestep", fontsize=12)
    ax2.set_ylabel("Feature Similarity", fontsize=12)
    ax2.set_title("Node Pair Similarity Over Time", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Initialize lines for tracked pairs
    timeline_lines = []
    time_points = np.arange(num_timesteps)
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_pairs)))

    for i, ((node1, node2), _) in enumerate(top_pairs):
        (line,) = ax2.plot(
            [],
            [],
            "-",
            label=f"Nodes {node1}↔{node2}",
            color=colors[i],
            linewidth=2,
            marker="o",
            markersize=4,
        )
        timeline_lines.append((line, node1, node2))

    ax2.legend(loc="upper right")

    # Initialize heatmap
    im = ax1.imshow(all_similarities[0], cmap="viridis", vmin=-1, vmax=1)
    ax1.set_title("Feature Similarity Matrix (t=0)")
    ax1.set_xlabel("Node Index")
    ax1.set_ylabel("Node Index")

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax1, shrink=0.6)
    cbar.set_label("Cosine Similarity")

    # Add time indicator
    time_text = fig.text(0.5, 0.95, "Timestep: 0", ha="center", fontsize=14)

    def update(frame):
        # Update heatmap
        im.set_array(all_similarities[frame])
        ax1.set_title(f"Feature Similarity Matrix (t={frame})")

        # Update time indicator
        time_text.set_text(f"Timestep: {frame}")

        # Update lines
        for line, node1, node2 in timeline_lines:
            similarity_values = [
                sim[node1, node2] for sim in all_similarities[: frame + 1]
            ]
            line.set_data(range(frame + 1), similarity_values)

        return [im, time_text] + [line for line, _, _ in timeline_lines]

    ani = FuncAnimation(fig, update, frames=num_timesteps, interval=200, blit=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for time indicator

    # Save animation
    ani.save("gcn_similarity_over_time.gif", writer="pillow", fps=5, dpi=100)

    return fig, ani


def visualize_gcn_relationships_static(
    model, state_sequence, num_samples=5, figsize=(18, 12)
):
    """
    Create a static visualization of GCN node relationships over time.

    Args:
        model: Your GCN model
        state_sequence: List of state tensors representing timesteps [t, batch, n_nodes, features]
        num_samples: Number of timesteps to sample for visualization
        figsize: Figure size (width, height)
    """

    num_timesteps = len(state_sequence)
    n_nodes = state_sequence[0].shape[1]

    # Sample evenly spaced timesteps
    if num_timesteps <= num_samples:
        sampled_timesteps = list(range(num_timesteps))
    else:
        sampled_timesteps = [
            int(i * (num_timesteps - 1) / (num_samples - 1)) for i in range(num_samples)
        ]

    # Storage for embeddings and similarities
    all_embeddings = []
    all_similarities = []

    # Process embeddings for each timestep
    with torch.no_grad():
        for t in sampled_timesteps:
            # Get graph for this timestep
            graph_list = model.create_chain_graph_batch(state_sequence[t])
            batched_graph = Batch.from_data_list(graph_list)

            # Get node embeddings
            embeddings = model.forward(batched_graph).cpu().numpy()
            all_embeddings.append(embeddings[:n_nodes])  # First graph only

            # Calculate similarity matrix
            graph_embeddings = embeddings[:n_nodes]
            sim_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    sim_matrix[i, j] = np.dot(
                        graph_embeddings[i], graph_embeddings[j]
                    ) / (
                        np.linalg.norm(graph_embeddings[i])
                        * np.linalg.norm(graph_embeddings[j])
                    )
            all_similarities.append(sim_matrix)

    # Create figure with multiple panels
    fig = plt.figure(figsize=figsize)

    # Define grid layout
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1])

    # 1. Top row: Similarity heatmaps at different timesteps
    for i, t_idx in enumerate(sampled_timesteps[:3]):  # Show first 3 timesteps
        ax = fig.add_subplot(gs[0, i])

        # Plot heatmap
        sns.heatmap(
            all_similarities[i],
            cmap="viridis",
            vmin=-1,
            vmax=1,
            annot=False,
            ax=ax,
            cbar=(i == 2),
        )
        ax.set_title(f"Timestep {t_idx}")
        ax.set_xlabel("Node Index")
        ax.set_ylabel("Node Index")

    # 2. Bottom left: Track specific node pairs over time
    ax_pairs = fig.add_subplot(gs[1, 0])

    # Find pairs with highest variance in similarity
    all_pairs = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if i != j:
                similarities = [sim[i, j] for sim in all_similarities]
                variance = np.var(similarities)
                all_pairs.append(((i, j), variance, similarities))

    # Sort and get top 5 pairs
    top_pairs = sorted(all_pairs, key=lambda x: x[1], reverse=True)[:5]

    # Plot similarity trends
    for (i, j), _, similarities in top_pairs:
        ax_pairs.plot(
            sampled_timesteps, similarities, marker="o", label=f"Nodes {i}↔{j}"
        )

    ax_pairs.set_xlabel("Timestep")
    ax_pairs.set_ylabel("Similarity")
    ax_pairs.set_title("Node Pair Similarity Over Time")
    ax_pairs.grid(True, alpha=0.3)
    ax_pairs.legend()

    # 3. Bottom middle: Embedding trajectory in 2D space
    ax_traj = fig.add_subplot(gs[1, 1])

    # Flatten and stack embeddings from all timesteps for PCA
    flattened_embeddings = np.vstack([emb for emb in all_embeddings])

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(flattened_embeddings)

    # Split back into timesteps
    reduced_by_time = np.split(reduced_embeddings, len(sampled_timesteps))

    # Plot trajectory for each node
    for node_idx in range(n_nodes):
        x_coords = [
            reduced_by_time[t][node_idx, 0] for t in range(len(sampled_timesteps))
        ]
        y_coords = [
            reduced_by_time[t][node_idx, 1] for t in range(len(sampled_timesteps))
        ]

        # Plot with arrow indicating direction
        ax_traj.plot(x_coords, y_coords, marker="o", label=f"Node {node_idx}")

        # Add arrows to show direction of movement
        for t in range(len(sampled_timesteps) - 1):
            ax_traj.annotate(
                "",
                xy=(x_coords[t + 1], y_coords[t + 1]),
                xytext=(x_coords[t], y_coords[t]),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="gray"),
            )

    ax_traj.set_title("Node Embedding Trajectories (PCA)")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()

    # 4. Bottom right: Similarity change heatmap (final - initial)
    ax_diff = fig.add_subplot(gs[1, 2])

    # Calculate difference between first and last similarity matrix
    sim_diff = all_similarities[-1] - all_similarities[0]

    # Plot difference heatmap
    sns.heatmap(sim_diff, cmap="coolwarm", center=0, annot=False, ax=ax_diff)
    ax_diff.set_title("Similarity Change (Final - Initial)")
    ax_diff.set_xlabel("Node Index")
    ax_diff.set_ylabel("Node Index")

    plt.tight_layout()
    plt.savefig("gcn_relationships_over_time.png", dpi=300, bbox_inches="tight")

    return fig


def visualize_attention_weights(model, state_batch):
    """Visualize which nodes the model pays attention to when making decisions"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Forward pass to get node embeddings
    with torch.no_grad():
        graph_list = model.create_chain_graph_batch(state_batch)
        batched_graph = Batch.from_data_list(graph_list)
        node_embeddings = model.forward(batched_graph)

        # Extract attention weights from attentional aggregation
        # We need to run the gate_nn inside AttentionalAggregation
        gate = model.att_pool.gate_nn(node_embeddings)
        weights = gate.squeeze(-1).softmax(dim=0).cpu().numpy()

    # Plot attention weights
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=np.arange(len(weights)), y=weights, ax=ax)
    ax.set_title("Attention Weights for Each Node")
    ax.set_xlabel("Node Index")
    ax.set_ylabel("Attention Weight")

    plt.tight_layout()
    return fig


def visualize_attention_weights_over_time(model, state_sequence, figsize=(12, 8)):
    """Visualize how attention weights change over time.

    Args:
        model: Your GCN model
        state_sequence: List of state tensors representing timesteps
        figsize: Size of the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    num_timesteps = len(state_sequence)
    n_nodes = state_sequence[0].shape[1]

    # Storage for attention weights over time
    attention_weights_over_time = []

    # Process each timestep
    with torch.no_grad():
        for t, state_batch in enumerate(state_sequence):
            # Get graph for this timestep
            graph_list = model.create_chain_graph_batch(state_batch)
            batched_graph = Batch.from_data_list(graph_list)

            # Forward pass to get node embeddings
            node_embeddings = model.forward(batched_graph)

            # Extract attention weights from attentional aggregation
            gate = model.att_pool.gate_nn(node_embeddings)
            weights = gate.squeeze(-1).softmax(dim=0).cpu().numpy()

            attention_weights_over_time.append(weights)

    # Convert to numpy array for easier handling
    attention_weights_over_time = np.array(attention_weights_over_time)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]}
    )

    # Plot 1: Attention weights over time as line plot
    # Create colormap to distinguish nodes better
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, n_nodes))

    for node_idx in range(n_nodes):
        ax1.plot(
            range(num_timesteps),
            attention_weights_over_time[:, node_idx],
            marker="o",
            linewidth=2,
            markersize=4,
            color=colors[node_idx],
            label=f"Node {node_idx}",
        )

    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Attention Weight")
    ax1.set_title("Attention Weights for Each Node Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Plot 2: Attention weights over time as heatmap
    sns.heatmap(
        attention_weights_over_time.T,  # Transpose to have nodes as rows, time as columns
        cmap="viridis",
        ax=ax2,
        cbar=True,
        xticklabels=5,  # Show every 5th timestep label
        yticklabels=np.arange(n_nodes),
    )
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Node")
    ax2.set_title("Attention Weight Heatmap")

    plt.tight_layout()
    plt.savefig("attention_weights_over_time.png", dpi=300)
    return fig


def visualize_edge_importance_over_time(
    model, state_sequence, node_idx=1, figsize=(14, 10)
):
    """
    Visualize how edge importance changes over time using GNNExplainer
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import seaborn as sns
    from torch_geometric.explain import Explainer, GNNExplainer

    num_timesteps = len(state_sequence)

    # Create a wrapper model that returns what GNNExplainer expects
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, edge_index):
            data = Data(x=x, edge_index=edge_index)
            batch = Batch.from_data_list([data])
            return self.model.forward(batch)

    wrapped_model = ModelWrapper(model)

    # Initialize GNNExplainer
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )

    # Storage for edge importance over time
    edge_importance_over_time = []
    edge_indices_list = []

    # IMPORTANT: Remove torch.no_grad() here since explainer needs gradients
    for t, state in enumerate(state_sequence):
        print(f"Processing timestep {t}...")

        # Create graph for this timestep - still use no_grad for this part
        with torch.no_grad():
            graph_list = model.create_chain_graph_batch(state)
            graph = graph_list[0]  # Use first graph

            # Get node features and edge index
            node_features = graph.x
            edge_index = graph.edge_index

            # Store edge indices for reference
            edge_indices_list.append(edge_index.cpu().numpy())

        # Create copies that require gradients
        node_features_grad = node_features.clone().detach().requires_grad_(True)
        edge_index_grad = (
            edge_index.clone().detach()
        )  # Edge indices typically don't need gradients

        # Run the explainer WITHOUT torch.no_grad()
        try:
            # Get explanations - we explain predictions for the selected node
            explanation = explainer(
                x=node_features_grad,
                edge_index=edge_index_grad,
                target=node_idx,
            )

            # Extract edge importance
            edge_mask = explanation.edge_mask.detach().cpu().numpy()
            edge_importance_over_time.append(edge_mask)
        except Exception as e:
            print(f"Error at timestep {t}: {e}")
            # If explanation fails, use zeros as fallback
            num_edges = edge_index.shape[1]
            edge_importance_over_time.append(np.zeros(num_edges))
            continue

    # Rest of the function remains the same...
    # Convert to numpy array
    edge_importance_over_time = np.array(edge_importance_over_time)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    # Get number of edges from first timestep
    num_edges = edge_importance_over_time[0].shape[0]

    # 1. Line plot for top edges over time
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate average importance for each edge
    avg_importance = np.mean(edge_importance_over_time, axis=0)

    # Get indices of top 5 edges based on average importance
    top_edge_indices = np.argsort(-avg_importance)[:5]

    # Plot top edges over time with distinct colors
    cmap = plt.cm.tab10
    colors = cmap(np.linspace(0, 1, len(top_edge_indices)))

    for i, edge_idx in enumerate(top_edge_indices):
        ax1.plot(
            range(num_timesteps),
            edge_importance_over_time[:, edge_idx],
            linewidth=2,
            marker="o",
            markersize=4,
            color=colors[i],
            label=f"Edge {edge_idx}",
        )

    # Reference edges from first timestep
    edges_ref = edge_indices_list[0]

    # Add edge annotations to the legend
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = []
    for i, edge_idx in enumerate(top_edge_indices):
        src, dst = edges_ref[0, edge_idx], edges_ref[1, edge_idx]
        new_labels.append(f"Edge {edge_idx} ({src.item()}->{dst.item()})")

    ax1.legend(handles, new_labels, loc="upper right")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Edge Importance")
    ax1.set_title(f"Top 5 Edge Importance Over Time for Node {node_idx}")
    ax1.grid(True, alpha=0.3)

    # 2. Heatmap of all edge importance over time
    ax2 = fig.add_subplot(gs[1, :])
    sns.heatmap(
        edge_importance_over_time.T,  # Transpose to have edges as rows, time as columns
        cmap="viridis",
        ax=ax2,
        cbar=True,
        xticklabels=5,  # Show every 5th timestep
    )
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Edge Index")
    ax2.set_title("All Edge Importance Over Time")

    # 3. Network visualization for first and last timestep
    t_first, t_last = 0, -1

    # First timestep network
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_network_with_edge_importance(
        edge_indices_list[t_first],
        edge_importance_over_time[t_first],
        ax=ax3,
        title=f"Edge Importance at t={t_first}",
    )

    # Last timestep network
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_network_with_edge_importance(
        edge_indices_list[t_last],
        edge_importance_over_time[t_last],
        ax=ax4,
        title=f"Edge Importance at t={num_timesteps-1}",
    )

    plt.tight_layout()
    plt.savefig(
        f"edge_importance_over_time_node{node_idx}.png", dpi=300, bbox_inches="tight"
    )
    return fig


def _plot_network_with_edge_importance(edge_index, edge_importance, ax, title):
    """Helper function to plot network with edge importance"""
    import networkx as nx

    # Create networkx graph
    G = nx.DiGraph()

    # Add nodes and edges
    edges = edge_index.T

    # Get number of nodes from edge index
    num_nodes = edge_index.max() + 1

    # Add nodes
    for i in range(num_nodes):
        G.add_node(i)

    # Add edges with importance weights
    for i, (src, dst) in enumerate(edges):
        src_idx, dst_idx = src.item(), dst.item()
        G.add_edge(src_idx, dst_idx, weight=edge_importance[i])

    # Position nodes in a circle layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, node_color="lightblue", node_size=500, alpha=0.8, ax=ax
    )

    # Create edge list with importance-based width
    edge_widths = [G[u][v]["weight"] * 5.0 for u, v in G.edges()]

    # Normalize edge widths
    if edge_widths:
        max_width = max(edge_widths)
        if max_width > 0:
            edge_widths = [w / max_width * 5.0 for w in edge_widths]

    # Draw edges with width based on importance
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        edge_color=edge_widths,
        edge_cmap=plt.cm.viridis,
        alpha=0.7,
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)

    ax.set_title(title)
    ax.axis("off")
