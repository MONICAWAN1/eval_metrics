import math

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


class CTClassifierNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self, x: Float[Tensor, "... {self.input_dim}"]
    ) -> Float[Tensor, "... {self.output_dim}"]:
        return self.net(x)


# --- Set Transformer attention blocks -------------------------------------------------
# Adapted from the official Set Transformer implementation,
# https://github.com/juho-lee/set_transformer (Lee et al., "Set Transformer", ICML 2019).
# We use Multihead Attention Blocks to pool the microenvironment instead of the symmetric
# mean/max DeepSet pooling, so the centroid is not diluted by its neighbours.


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
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
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


# --------------------------------------------------------------------------------------


class SpatialCTClassifierBase(nn.Module):
    """Base for microenvironment (spatial) cell-type classifiers.

    Input is the ``[gene_expression | relative_position]`` point set with the centroid at
    point 0 (see :class:`~nicheflow.datasets.h5ad_ct_dataset.SpatialH5ADCTDataset` and
    :func:`~nicheflow.tasks.flow_matching.build_microenv_points`). With
    ``mask_centroid=True`` (default) the centroid is excluded so the prediction uses the
    *neighbourhood only* — the metric scores spatial coherence and cannot take the
    ``x0 -> type`` shortcut (no leak of any center-derived feature).

    Subclasses keep ``input_dim``/``output_dim`` and the ``"X"`` batch key, so the
    :class:`~nicheflow.tasks.ct_classification.CellTypeClassification` task and the
    :class:`~nicheflow.tasks.flow_matching.CellTypeConcordance` eval (which read
    ``net.output_dim`` and call ``net(batch["X"])``) work unchanged across variants.
    ``FlowMatching`` recognises any subclass as a spatial classifier.
    """

    def __init__(
        self, input_dim: int, output_dim: int, coord_dim: int = 2, mask_centroid: bool = True
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.coord_dim = coord_dim
        self.output_dim = output_dim
        self.mask_centroid = mask_centroid


class SpatialCTClassifierNet(SpatialCTClassifierBase):
    """Set-Transformer classifier over a microenvironment point set.

    Input is a per-cell point cloud of ``[gene_expression | relative_position]`` rows
    (the centroid plus its nearest neighbours, see
    :class:`~nicheflow.datasets.h5ad_ct_dataset.SpatialH5ADCTDataset`). The centroid is
    the first point (relative position 0) on both the training and the eval
    (:func:`~nicheflow.tasks.flow_matching.build_microenv_points`) paths.

    By default (``mask_centroid=True``) the centroid is dropped from the input and the
    type is predicted from the neighbours only, so the metric measures whether a cell
    sits in a spatially coherent niche. The neighbours are embedded per-point, mixed by 
    a ``SAB`` self-attention stack, and pooled by a learnable ``PMA`` seed. Relative 
    positions keep the model translation invariant.

    With ``mask_centroid=False`` (ablation only) the centroid is kept and used as the
    attention query (``MAB``) over its neighbours, then concatenated with its own
    embedding before decoding. This intentionally re-introduces the ``x0`` leak, so it is
    only useful for measuring how much spatial context adds on top of expression.

    Both modes use the attention blocks of the Set Transformer (Lee et al., ICML 2019).

    The output is still cell-type logits and the batch key is still ``"X"``, so the
    existing :class:`~nicheflow.tasks.ct_classification.CellTypeClassification` task and
    the :class:`~nicheflow.tasks.flow_matching.CellTypeConcordance` eval (which read
    ``net.output_dim`` and call ``net(batch["X"])``) work unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        coord_dim: int = 2,
        num_heads: int = 4,
        num_sabs: int = 2,
        mask_centroid: bool = True,
        ln: bool = True,
    ) -> None:
        super().__init__(input_dim, output_dim, coord_dim, mask_centroid)
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )
        point_dim = input_dim + coord_dim

        # Shared per-point embedding of each [expression | relative_position] row.
        self.point_proj = nn.Linear(point_dim, hidden_dim)
        # Let the neighbours interact (captures local niche structure).
        self.encoder = nn.Sequential(
            *[SAB(hidden_dim, hidden_dim, num_heads, ln=ln) for _ in range(num_sabs)]
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
            nn.Linear(dec_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self, x: Float[Tensor, "batch n_points {self.input_dim}+{self.coord_dim}"]
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


# class GraphCTClassifierNet(SpatialCTClassifierBase):
#     """GCN node classifier over a microenvironment graph (the simplest GNN probe).

#     Uses the GNN-native data representation (Kipf & Welling, ICLR 2017; see also the GCN
#     cell-type annotators scIMGCN and STdGCN): gene expression is the **node feature**
#     (*what* a cell is) and the spatial coordinates define the **graph structure** (*who is
#     near whom*) — they are NOT concatenated into the feature vector. Per microenvironment
#     we build a kNN graph among the centroid + its neighbours from their relative positions,
#     run ``GCNConv`` message passing, and read out the centroid node's embedding.

#     Masked centroid (default): the centroid node's expression is zeroed **and** its
#     position is forced to the origin, so nothing from the centre can leak into its own
#     prediction — its type is predicted purely from neighbours' expression propagated along
#     the spatial graph. The graph is built *within a niche* (the centroid's neighbourhood),
#     not over the whole pool of generated cells, matching the receptive field of the other
#     spatial probes.

#     Input is the same ``[gene_expression | relative_position]`` point set as the other
#     spatial classifiers (centroid first), so this is a drop-in for the
#     :class:`~nicheflow.tasks.ct_classification.CellTypeClassification` task and the
#     :class:`~nicheflow.tasks.flow_matching.CellTypeConcordance` eval.
#     """

#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         hidden_dim: int,
#         coord_dim: int = 2,
#         mask_centroid: bool = True,
#         num_layers: int = 2,
#         intra_k: int = 8,
#         dropout: float = 0.5,
#     ) -> None:
#         super().__init__(input_dim, output_dim, coord_dim, mask_centroid)
#         self.intra_k = intra_k
#         self.dropout = dropout
#         # Node features = gene expression only (coordinates define the graph, not features).
#         self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])
#         for _ in range(num_layers - 1):
#             self.convs.append(GCNConv(hidden_dim, hidden_dim))
#         self.head = nn.Linear(hidden_dim, output_dim)

#     def forward(
#         self, x: Float[Tensor, "batch n_points {self.input_dim}+{self.coord_dim}"]
#     ) -> Float[Tensor, "batch {self.output_dim}"]:
#         batch_size, n_points, _ = x.shape
#         feats = x[..., : self.input_dim]  # (B, K, input_dim) gene expression
#         pos = x[..., self.input_dim : self.input_dim + self.coord_dim]  # (B, K, coord_dim)

#         if self.mask_centroid:
#             # Nothing from the centre may leak: zero its expression and pin it to the
#             # origin (its relative position is 0 by construction anyway).
#             feats = feats.clone()
#             pos = pos.clone()
#             feats[:, 0, :] = 0.0
#             pos[:, 0, :] = 0.0

#         # One block-diagonal batched graph: a per-niche kNN over relative positions.
#         edge_index = self._niche_knn_edges(pos)
#         h = feats.reshape(batch_size * n_points, self.input_dim)
#         for i, conv in enumerate(self.convs):
#             h = conv(h, edge_index)
#             if i < len(self.convs) - 1:
#                 h = F.dropout(F.relu(h), p=self.dropout, training=self.training)

#         # Read out the centroid node of each niche (row 0 within each K-node block).
#         centroid_rows = torch.arange(batch_size, device=x.device) * n_points
#         return self.head(h[centroid_rows])

#     def _niche_knn_edges(self, pos: Tensor) -> Tensor:
#         """Undirected kNN edges within each niche, offset into one batched graph."""
#         batch_size, n_points, _ = pos.shape
#         k = min(self.intra_k, n_points - 1)
#         dist = torch.cdist(pos, pos)  # (B, K, K)
#         eye = torch.eye(n_points, dtype=torch.bool, device=pos.device)
#         dist = dist.masked_fill(eye, float("inf"))  # exclude self
#         nbr = dist.topk(k, dim=-1, largest=False).indices  # (B, K, k)

#         offset = (torch.arange(batch_size, device=pos.device) * n_points).view(-1, 1, 1)
#         src = (torch.arange(n_points, device=pos.device).view(1, -1, 1).expand(batch_size, -1, k) + offset)
#         dst = nbr + offset
#         edge_index = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)
#         return to_undirected(edge_index, num_nodes=batch_size * n_points)
