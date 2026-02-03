# Re-running after kernel reset: define functions and demo again.

import numpy as np
import matplotlib.pyplot as plt
import os


def _clean_1d(arr):
    arr = np.asarray(arr, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def freedman_diaconis_bins(data: np.ndarray) -> int:
    data = _clean_1d(data)
    n = data.size
    if n < 2:
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = np.std(data) * 1.349
    h = 2.0 * iqr * (n ** (-1 / 3))
    if h <= 0:
        return max(1, int(np.sqrt(max(n, 1))))
    bins = int(np.ceil((data.max() - data.min()) / h))
    return max(1, bins)


def gaussian_kde_1d(data: np.ndarray, grid: np.ndarray = None):
    data = _clean_1d(data)
    n = data.size
    if n == 0:
        raise ValueError("Empty data for KDE")
    std = np.std(data, ddof=1) if n > 1 else 1.0
    bw = 1.06 * std * (n ** (-1 / 5))
    if bw <= 0:
        bw = 1e-3
    if grid is None:
        pad = 3.0 * bw
        grid = np.linspace(data.min() - pad, data.max() + pad, 512)
    diffs = (grid[:, None] - data[None, :]) / bw
    kernel_vals = np.exp(-0.5 * diffs**2) / np.sqrt(2 * np.pi)
    density = kernel_vals.mean(axis=1) / bw
    return grid, density, bw


def ecdf(data: np.ndarray):
    data = _clean_1d(data)
    if data.size == 0:
        return np.array([]), np.array([])
    x = np.sort(data)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def _maybe_subsample_T(X, max_points=None):
    if max_points is None:
        return X
    T = X.shape[0]
    if T <= max_points:
        return X
    idx = np.linspace(0, T - 1, max_points).astype(int)
    return X[idx]


def plot_hist_kde_per_dim(X, dims=None, max_points=None):
    X = np.asarray(X)
    assert X.ndim == 3, "X should be shaped (T, A, D)"
    Xs = _maybe_subsample_T(X, max_points)
    T, A, D = Xs.shape
    if dims is None:
        dims = list(range(D))
    for d in dims:
        data = _clean_1d(Xs[..., d])
        if data.size == 0:
            continue
        bins = freedman_diaconis_bins(data)
        plt.figure()
        plt.hist(data, bins=bins, density=True)
        plt.title(f"Dim {d}: Histogram (density) with FD bins ({bins})")
        plt.xlabel(f"obs[{d}]")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()

        grid, dens, bw = gaussian_kde_1d(data)
        plt.figure()
        plt.plot(grid, dens)
        plt.title(f"Dim {d}: Gaussian KDE (bw≈{bw:.3g})")
        plt.xlabel(f"obs[{d}]")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()


def plot_ecdf_dim(X, dim, max_points=None):
    Xs = _maybe_subsample_T(np.asarray(X), max_points)
    data = _clean_1d(Xs[..., dim])
    x, y = ecdf(data)
    plt.figure()
    plt.step(x, y, where="post")
    plt.title(f"Dim {dim}: Empirical CDF")
    plt.xlabel(f"obs[{dim}]")
    plt.ylabel("F(x)")
    plt.tight_layout()
    plt.show()


def plot_agent_overlays(
    X, dim, agent_ids=None, max_points=None, episode=0, save_path=None
):
    Xs = _maybe_subsample_T(np.asarray(X), max_points)
    T, A, D = Xs.shape
    if agent_ids is None:
        agent_ids = list(range(A))
    plt.figure()
    for a in agent_ids:
        dat = _clean_1d(Xs[:, a, dim])
        if dat.size < 5:
            continue
        grid, dens, _ = gaussian_kde_1d(dat)
        plt.plot(grid, dens, alpha=0.8, label=f"agent {a}")
    plt.title(f"Dim {dim}: Per-agent KDE overlays")
    plt.xlabel(f"obs[{dim}]")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(
        os.path.dirname(save_path), f"obsevation_dist_{episode}_{dim}.png"
    )
    plt.savefig(path, dpi=300)


def plot_hexbin_dims(X, dim_x, dim_y, max_points=None):
    Xs = _maybe_subsample_T(np.asarray(X), max_points)
    x = _clean_1d(Xs[..., dim_x])
    y = _clean_1d(Xs[..., dim_y])
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    plt.figure()
    hb = plt.hexbin(x, y, gridsize=40)
    plt.title(f"Hexbin: obs[{dim_x}] vs obs[{dim_y}]")
    plt.xlabel(f"obs[{dim_x}]")
    plt.ylabel(f"obs[{dim_y}]")
    cb = plt.colorbar(hb)
    cb.set_label("Counts")
    plt.tight_layout()
    plt.show()


def summary_stats_per_dim(X, max_points=None):
    Xs = _maybe_subsample_T(np.asarray(X), max_points)
    D = Xs.shape[2]
    stats = []
    for d in range(D):
        data = _clean_1d(Xs[..., d])
        if data.size == 0:
            stats.append((d, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        stats.append(
            (
                d,
                data.size,
                float(np.mean(data)),
                float(np.std(data, ddof=1)) if data.size > 1 else 0.0,
                float(np.min(data)),
                float(np.percentile(data, 25)),
                float(np.median(data)),
                float(np.percentile(data, 75)),
                float(np.max(data)),
            )
        )
    import pandas as pd

    df = pd.DataFrame(
        stats,
        columns=["dim", "count", "mean", "std", "min", "q25", "median", "q75", "max"],
    )
    return df


# ---- Demo with synthetic data ----
# rng = np.random.default_rng(123)
# T, A, D = 2000, 6, 8
# base = rng.normal(size=(T, 1, D))
# agent_bias = rng.normal(scale=0.5, size=(1, A, D))
# trend = np.linspace(0, 1.0, T).reshape(T, 1, 1) * rng.normal(scale=0.3, size=(1, 1, D))
# X = base + agent_bias + trend

# # Run a few example plots
# plot_hist_kde_per_dim(X, dims=[0, 1], max_points=800)
# plot_ecdf_dim(X, dim=0, max_points=800)
# plot_agent_overlays(X, dim=0, agent_ids=[0, 1, 2], max_points=800)
# plot_hexbin_dims(X, dim_x=0, dim_y=1, max_points=800)
