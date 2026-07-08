import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class CTClassifierNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: Float[Tensor, "... {self.input_dim}"],
    ) -> Float[Tensor, "... {self.output_dim}"]:
        return self.net(x)


# --- Set Transformer attention blocks -------------------------------------------------
# Adapted from the official Set Transformer implementation,
# https://github.com/juho-lee/set_transformer (Lee et al., "Set Transformer", ICML 2019).
# We use Multihead Attention Blocks to pool the microenvironment instead of the symmetric
# mean/max DeepSet pooling, so the centroid is not diluted by its neighbours.


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# --------------------------------------------------------------------------------------


class SpatialCTClassifierBase(nn.Module):
    """Base for microenvironment (spatial) cell-type classifiers.

    Input is an **expression-only** point set ``(batch, k+1, input_dim)`` — the centroid (point 0)
    plus its ``k`` nearest neighbours (KNN by coordinates; see
    :class:`~paired_slides_eval.classifier.dataset.SpatialH5ADCTDataset` and
    :func:`~paired_slides_eval.metrics._common.build_knn_point_set`). **Coordinates are not
    features** — they are used only to select KNN membership, then discarded, so the classifier is
    coordinate-blind (rotation/translation invariant). Local spatial organisation is evaluated
    *implicitly*: a realistic generated cell should have a neighbourhood-expression signature like its
    nearest real cell's. With ``mask_centroid=True`` (default) the centroid is excluded so the
    prediction uses the *neighbourhood only* (no ``x0 -> type`` leak).

    Subclasses keep ``input_dim``/``output_dim`` and the ``"X"`` batch key, so the
    :class:`~paired_slides_eval.classifier.task.CellTypeClassification` task and the eval metrics
    (which read ``net.output_dim`` and call ``net(batch["X"])``) work unchanged across variants.

    """

    def __init__(self, input_dim: int, output_dim: int, mask_centroid: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_centroid = mask_centroid


class SpatialCTClassifierNet(SpatialCTClassifierBase):
    """Set-Transformer classifier over an **expression-only** KNN neighbourhood.

    Input is a per-cell point set of **gene-expression** rows ``(batch, k+1, input_dim)`` — the
    centroid (point 0) plus its ``k`` nearest neighbours (KNN by coordinates; see
    :class:`~paired_slides_eval.classifier.dataset.SpatialH5ADCTDataset`). Coordinates are **not**
    fed in: they only define which cells are in the set, so the classifier is coordinate-blind and
    cannot exploit a relative-position leak (which would not survive rotation anyway).

    By default (``mask_centroid=True``) the centroid is dropped and the type is predicted from the
    neighbours only, so the metric measures whether a cell sits in a coherent local neighbourhood.
    The neighbours are embedded per-point, mixed by a ``SAB`` self-attention stack, and pooled by a
    learnable ``PMA`` seed. With ``mask_centroid=False`` (ablation) the centroid embedding queries its
    neighbours (``MAB``) and is concatenated back — re-introducing the ``x0`` leak.

    Both modes use the attention blocks of the Set Transformer (Lee et al., ICML 2019). The output is
    cell-type logits and the batch key is ``"X"``, so the
    :class:`~paired_slides_eval.classifier.task.CellTypeClassification` task and the eval metrics work
    unchanged.

    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_sabs: int = 2,
        mask_centroid: bool = True,
        ln: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, mask_centroid)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).",
            )

        # Shared per-point embedding of each gene-expression row (coordinates are not features).
        self.point_proj = nn.Linear(input_dim, hidden_dim)
        # Let the neighbours interact (captures local niche structure).
        self.encoder = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads, ln=ln) for _ in range(num_sabs)],
        )

        if mask_centroid:
            # Masked centroid: pool the neighbours-only set with a learnable seed.
            self.pool = PMA(hidden_dim, num_heads, num_seeds=1, ln=ln)
            dec_in = hidden_dim
        else:
            # Centroid-conditioned: the centroid embedding queries its neighbours, then
            # we concat it with the attended neighbourhood (so its own signal is kept).
            self.pool = MAB(hidden_dim, hidden_dim, hidden_dim, num_heads, ln=ln)
            dec_in = 2 * hidden_dim

        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: Float[Tensor, "batch n_points {self.input_dim}"],
    ) -> Float[Tensor, "batch {self.output_dim}"]:
        h = self.point_proj(x)  # (batch, n_points, hidden_dim)

        if self.mask_centroid:
            neighbors = self.encoder(h[:, 1:, :])  # drop the centroid (point 0)
            pooled = self.pool(neighbors)  # (batch, 1, hidden_dim)
            return self.dec(pooled.squeeze(1))

        centroid = h[:, :1, :]  # (batch, 1, hidden_dim)
        neighbors = self.encoder(h[:, 1:, :])  # (batch, n_points - 1, hidden_dim)
        attended = self.pool(centroid, neighbors)  # centroid attends neighbours
        feats = torch.cat([centroid.squeeze(1), attended.squeeze(1)], dim=-1)
        return self.dec(feats)


# class SpatialDeepSetCTClassifierNet(SpatialCTClassifierBase):
#     """DeepSet spatial classifier (shared per-point MLP + symmetric mean/max pooling).

#     This is the upstream-style architecture, kept as a baseline to compare head-to-head
#     with the attention-based :class:`SpatialCTClassifierNet`. Both mask the centroid by
#     default, so both score pure spatial-context coherence (no ``x0 -> type`` leak); the
#     only difference is the pooling — symmetric mean+max here vs. centroid/seed attention
#     in the Set-Transformer variant.

#     Each neighbour point is embedded by ``phi``; the set is pooled with mean+max (making
#     the model permutation invariant over the neighbours) and decoded to cell-type logits
#     by ``rho``. With ``mask_centroid=True`` the centroid (point 0) is dropped before
#     pooling. Relative positions keep the model translation invariant.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         hidden_dim: int,
#         coord_dim: int = 2,
#         mask_centroid: bool = True,
#     ) -> None:
#         super().__init__(input_dim, output_dim, coord_dim, mask_centroid)
#         point_dim = input_dim + coord_dim
#         self.phi = nn.Sequential(
#             nn.Linear(point_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.rho = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#         )

#     def forward(
#         self, x: Float[Tensor, "... n_points {self.input_dim}+{self.coord_dim}"]
#     ) -> Float[Tensor, "... {self.output_dim}"]:
#         points = x[..., 1:, :] if self.mask_centroid else x  # drop the centroid (point 0)
#         h = self.phi(points)
#         pooled = torch.cat([h.mean(dim=-2), h.max(dim=-2).values], dim=-1)
#         return self.rho(pooled)
