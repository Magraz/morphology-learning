import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot_pickle_data(filepath, save_dir=None):
    """
    Plot all plottable values from the pickle file.

    Args:
        filepath: Path to the pickle file
        save_dir: Directory to save plots (if None, will display plots)
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print("Data is not a dictionary. Cannot plot.")
        return

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Found {len(data.keys())} keys in the pickle file")

    # Plot each key in the dictionary
    for key, value in data.items():
        try:
            # Convert to list if numpy array
            if hasattr(value, "tolist"):
                value = value.tolist()

            # Skip if not plottable
            if not isinstance(value, (list, np.ndarray)) or len(value) == 0:
                print(f"Skipping {key}: Not plottable data type or empty")
                continue

            # Check if values are numeric
            if not all(isinstance(x, (int, float)) for x in value):
                print(f"Skipping {key}: Contains non-numeric values")
                continue

            # Create figure
            plt.figure(figsize=(10, 6))

            # Plot data with appropriate style
            if key.startswith("agent") and "_loss" in key:
                plt.semilogy(value, label=key)  # Log scale for loss values
            else:
                plt.plot(value, label=key)

            # Add labels and title
            plt.xlabel("Steps/Updates")
            plt.ylabel("Value")
            plt.title(f"Training Curve: {key}")
            plt.grid(True, alpha=0.3)

            # Add legend if useful
            if len(value) > 0:
                plt.legend()

            # Either save or show the plot
            if save_dir:
                # Clean up filename
                safe_key = key.replace("/", "_").replace("\\", "_")
                plt.savefig(os.path.join(save_dir, f"{safe_key}.png"), dpi=150)
                plt.close()
                print(f"Saved plot for {key}")
            else:
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Error plotting {key}: {e}")

    print("Plotting completed!")


def print_pickle_file(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

        # If data is a dictionary, print only the keys
        if isinstance(data, dict):
            print("Keys in the pickle file:")
            for key in data.keys():
                print(f"- {key}")

            # Optionally, print the shape/size of values if they're numpy arrays or lists
            print("\nValue shapes/sizes:")
            for key, value in data.items():
                if hasattr(value, "shape"):  # For numpy arrays
                    print(f"- {key}: shape {value.shape}")
                elif isinstance(value, list):
                    print(f"- {key}: list of length {len(value)}")
                else:
                    print(f"- {key}: {type(value)}")
        else:
            # Handle non-dictionary data
            print("Data is not a dictionary. Type:", type(data))


if __name__ == "__main__":
    # File path
    file_path = "/home/magraz/research/src/learning/experiments/results/mpe_spread_test/mlp_shared/debug/logs/training_stats_checkpoint.pkl"

    # Print file keys
    print_pickle_file(file_path)

    # Plot all data
    plots_dir = "/home/magraz/research/src/learning/experiments/results/mpe_spread_test/mlp_shared/debug/plots"
    plot_pickle_data(file_path, save_dir=plots_dir)

    print(f"\nPlots saved to: {plots_dir}")
