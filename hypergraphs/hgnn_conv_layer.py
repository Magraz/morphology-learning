import dhg
import torch
import torch.nn as nn


def smoothing_with_hgnn_factors(X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
    """Apply HGNN smoothing from incidence matrices only.

    This is algebraically equivalent to ``hg.smoothing_with_HGNN(X)``:

        D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X

    For this project's hypergraphs all hyperedge weights are unit weights, so
    ``W_e`` is the identity and we can derive the needed degree scalings
    directly from the incidence matrix. This avoids DHG's lazy ``W_e`` and
    ``L_HGNN`` construction paths, which were failing in batched critic runs.
    """
    if hg.device != X.device:
        X = X.to(hg.device)

    H = hg.H.coalesce()
    H_T = hg.H_T.coalesce()

    # With unit hyperedge weights, vertex degree is the row-sum of H and
    # hyperedge degree is the row-sum of H^T.
    d_v = torch.sparse.sum(H, dim=1).to_dense().clamp_min(1.0)
    d_e = torch.sparse.sum(H_T, dim=1).to_dense().clamp_min(1.0)

    X = X * d_v.rsqrt().unsqueeze(-1)
    X = torch.sparse.mm(H_T, X)
    X = X * d_e.reciprocal().unsqueeze(-1)
    X = torch.sparse.mm(H, X)
    return X * d_v.rsqrt().unsqueeze(-1)


class HGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        drop_rate: float = 0.5,
    ):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        X = self.theta(X)
        X_ = smoothing_with_hgnn_factors(X, hg)
        X_ = self.drop(self.act(X_))
        return X_
