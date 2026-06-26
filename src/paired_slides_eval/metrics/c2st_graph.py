"""Graph classifier two-sample test (graph-C2ST) — a spatially-aware C2ST.

The plain :mod:`~paired_slides_eval.metrics.c2st` trains an MLP on per-cell ``[expression | absolute
pos]``: every cell is judged in isolation, so the test mostly separates on absolute coordinates /
per-cell features and is blind to whether the *spatial arrangement* of real vs generated cells
matches. This variant replaces the MLP with a **simple GCN** (Kipf & Welling 2017): positions enter
only as a spatial **k-nearest-neighbour graph**, so the discriminator reads each cell's *relative*
neighbourhood rather than its absolute coordinates.

Design (a node-level two-sample test):

* Build **one joint** spatial kNN graph over the **combined** real+generated cloud. Mixing both
  labels in every neighbourhood is deliberate: separate per-slide graphs are disconnected
  components, and a GNN can then encode each component's mean — which differs between two finite
  samples even from the *same* distribution — making "which component am I in" a label shortcut that
  inflates the null far above 0.5. A joint graph removes that shortcut; a cell is judged by its own
  expression in the context of its real+generated spatial neighbours.
* **Node features = expression only.** Geometry enters purely through the graph topology, so the net
  cannot read absolute coordinates — only "does this cell's expression fit its spatial
  neighbourhood the way a real cell's would".
* A 2-layer GNN with **separate self + neighbour transforms** (GraphSAGE-style mean aggregation, not
  plain Kipf ``Â·H·W``) message-passes, then classifies **each node** real(0) vs generated(1). The
  self path preserves the cell's own expression (so a per-cell difference stays detectable); the
  neighbour path adds context, and together they can represent the ``expr - neighbourhood_mean``
  residual that flags a wrong spatial arrangement — a signal a pure Kipf GCN over-smooths away.
  Held-out node accuracy/AUC over ``n_folds`` stratified folds is the statistic: ~0.5
  indistinguishable (good), ~1.0 trivially separable (bad) — same reading as the MLP C2ST.

Self-contained pure-PyTorch (no ``torch_geometric``) so the group stays in the always-on core
suite. Both samples are subsampled to ``max_n`` (the dense adjacency is then at most ``2*max_n``
square), keeping the two-sample comparison fair and the cost bounded.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _normalized_knn_adjacency(pos: np.ndarray, k: int) -> np.ndarray:
    """Symmetric-normalized adjacency ``D^{-1/2}(A+I)D^{-1/2}`` of an undirected spatial kNN graph.

    ``A`` connects each cell to its ``k`` nearest neighbours (by Euclidean coords), symmetrized to
    undirected; self-loops are added (the ``+I`` in Kipf & Welling). Returns a dense ``(n, n)``
    matrix — fine for the subsampled clouds this metric uses.
    """
    pos = np.asarray(pos, dtype=np.float64)
    n = len(pos)
    k = int(min(k, n - 1))
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k + 1)  # column 0 is the cell itself
    idx = np.atleast_2d(idx)

    adj = np.zeros((n, n), dtype=np.float32)
    rows = np.repeat(np.arange(n), idx.shape[1])
    adj[rows, idx.reshape(-1)] = 1.0
    adj = np.maximum(adj, adj.T)  # undirected
    np.fill_diagonal(adj, 1.0)  # self-loops (A + I)

    deg = adj.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    return (adj * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]


def c2st_graph(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    graph_k: int = 10,
    n_folds: int = 5,
    hidden: int = 64,
    epochs: int = 100,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    dropout: float = 0.5,
    z_score: bool = True,
    seed: int = 0,
) -> tuple[float, float]:
    """Return ``(accuracy, roc_auc)`` of a 2-layer GCN separating real (0) from generated (1) cells.

    Nodes are cells with **expression** features; one joint spatial kNN graph (``graph_k``
    neighbours over the combined real+generated cloud) is how relative position enters. Both metrics
    are the mean over ``n_folds`` stratified node folds. ``real_x`` is the reference for z-scoring
    (pass the *real* sample), mirroring :func:`~paired_slides_eval.metrics.c2st.c2st`.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    real_x = np.asarray(real_x, dtype=np.float64)
    gen_x = np.asarray(gen_x, dtype=np.float64)
    if z_score:
        mean = real_x.mean(axis=0)
        std = real_x.std(axis=0) + 1e-8
        real_x = (real_x - mean) / std
        gen_x = (gen_x - mean) / std

    n_real, n_gen = len(real_x), len(gen_x)

    # ONE joint kNN graph over the combined cloud (real + generated). Mixing both labels in every
    # neighbourhood is deliberate: separate per-slide graphs would be disconnected components, and a
    # GCN can then encode each component's mean — which differs between two finite samples even from
    # the *same* distribution — turning "which component am I in" into a label shortcut that inflates
    # the null far above 0.5. With a joint graph there is no such shortcut: a cell is judged by its
    # own expression in the context of its real+generated spatial neighbours.
    all_pos = np.concatenate([np.asarray(real_pos), np.asarray(gen_pos)], axis=0)
    adj = _normalized_knn_adjacency(all_pos, graph_k)

    feats = np.concatenate([real_x, gen_x], axis=0).astype(np.float32)
    labels = np.concatenate([np.zeros(n_real), np.ones(n_gen)]).astype(np.int64)
    n_total = n_real + n_gen

    a_hat = torch.from_numpy(adj)
    x_t = torch.from_numpy(feats)
    y_t = torch.from_numpy(labels)

    class _GCN(nn.Module):
        """2-layer GNN with separate self + neighbour transforms (GraphSAGE-style mean aggregation).

        Each layer is ``relu(W_self·h + W_neigh·(Â·h))``: keeping a dedicated *self* path preserves
        the cell's own expression (so a real per-cell difference is still detectable), while the
        *neighbour* path adds spatial context. The relu over both lets the net represent the
        ``h - neighbourhood_mean`` residual — "does this cell's expression fit where it sits?" — which
        is what flags a wrong spatial arrangement. A plain Kipf ``Â·H·W`` (self folded into the
        averaged adjacency) over-smooths on the joint graph and washes that signal out.
        """

        def __init__(self, in_dim: int) -> None:
            super().__init__()
            self.self1 = nn.Linear(in_dim, hidden)
            self.neigh1 = nn.Linear(in_dim, hidden)
            self.self2 = nn.Linear(hidden, hidden)
            self.neigh2 = nn.Linear(hidden, hidden)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.self1(x) + self.neigh1(a @ x))
            h = F.dropout(h, dropout, training=self.training)
            h = F.relu(self.self2(h) + self.neigh2(a @ h))
            return self.head(h)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(n_total), labels)):
        torch.manual_seed(seed + fold)
        model = _GCN(feats.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_t = torch.from_numpy(train_idx)

        # Transductive node classification: message passing sees the whole (fixed) graph each
        # epoch; the loss is taken only on the training nodes.
        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(x_t, a_hat)
            loss = F.cross_entropy(logits[train_t], y_t[train_t])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(x_t, a_hat), dim=1)[:, 1].numpy()
        pred = (prob[test_idx] >= 0.5).astype(int)
        accs.append(float((pred == labels[test_idx]).mean()))
        aucs.append(float(roc_auc_score(labels[test_idx], prob[test_idx])))

    return float(np.mean(accs)), float(np.mean(aucs))


def c2st_graph_metrics(
    real_x: np.ndarray,
    real_pos: np.ndarray,
    gen_x: np.ndarray,
    gen_pos: np.ndarray,
    *,
    prefix: str = "",
    max_n: int = 2000,
    graph_k: int = 10,
    n_folds: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """Spatially-aware C2ST: a GCN over per-slide spatial kNN graphs, expression as node features.

    Complements the MLP :func:`~paired_slides_eval.metrics.c2st.c2st_metrics`: where that flattens
    cells and reads absolute coordinates, this judges each cell from its *relative* spatial
    neighbourhood. Reports ``c2st/graph_acc`` and ``c2st/graph_auc`` (~0.5 indistinguishable, ~1.0
    trivially separable). Real and generated clouds are each subsampled to ``max_n`` (seeded) so the
    comparison is fair and the dense graph stays bounded.
    """
    p = f"{prefix}/" if prefix else ""
    rng = np.random.default_rng(seed)

    def _sub(x: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x, pos = np.asarray(x), np.asarray(pos)
        if len(x) > max_n:
            sel = rng.choice(len(x), max_n, replace=False)
            return x[sel], pos[sel]
        return x, pos

    rx, rp = _sub(real_x, real_pos)
    gx, gp = _sub(gen_x, gen_pos)
    acc, auc = c2st_graph(rx, rp, gx, gp, graph_k=graph_k, n_folds=n_folds, seed=seed)
    return {f"{p}c2st/graph_acc": acc, f"{p}c2st/graph_auc": auc}
