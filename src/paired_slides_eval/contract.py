"""The input contract for evaluation: a real ``TargetSlide`` and the model's ``GeneratedNiches``.

The metrics take the real target slide and the generated cells — both as plain arrays. The
user-facing inputs are **original AnnData (``.h5ad``) files** (raw gene expression + spatial
coordinates); build the dataclasses with :meth:`TargetSlide.from_anndata` /
:meth:`GeneratedSlide.from_anndata` / :meth:`GeneratedNiches.from_anndata`.

Two shapes for the generated cells:

* :class:`GeneratedSlide` — a **flat** slide, ``x (N, D)`` + ``pos (N, D)``. Use this for
  whole-slide generative models. The label-free metrics (psd, spd, distribution, c2st, moran)
  run on it directly; the classifier metrics (concordance, ct_gap) are still computed by
  reconstructing niches from geometry (see :func:`paired_slides_eval.evaluate.evaluate`); only
  regression — which needs cell-for-cell matched ground truth — stays skipped.
* :class:`GeneratedNiches` — **niche-shaped**, ``x (B, N, D)`` (centroid at point 0). Required by
  the regression metric and the preferred input for the classifier metrics, which compare each
  generated niche to its model-supplied paired real microenvironment.

Conventions
-----------
* Expression and generated cells must share one feature space. The simplest way to guarantee
  this is to keep both as raw genes (same gene panel), or to fit one PCA on the target
  (``TargetSlide.from_anndata(..., n_pcs=50)``) and project the generated cells through it with
  ``.project(target.pca)``.
* A niche = a centroid cell (point index 0) + its ``N - 1`` neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paired_slides_eval.data.anndata import (
    cell_type_labels,
    fit_pca,
    read_anndata,
    slide_coords,
    slide_expression,
)


def _pca_aware_transform(x2d: np.ndarray, pca) -> np.ndarray:
    """Project ``x2d`` (rows = cells) through ``pca``, or pass it through if already PCA-reduced.

    The space is detected by **feature dimension**: an array whose width matches the PCA's input
    (raw-feature) dimension is in gene space and gets projected; one whose width matches the PCA's
    output (component) dimension is taken to be already in that PCA basis and is returned unchanged
    (so cells generated directly in PCA space — e.g. a flow-matching model — are not double-
    transformed). Any other width is an error: the cells share neither space with the target.

    Detection is dimensional only — when both sides are already reduced it cannot verify they came
    from the *same* PCA fit, so a pre-reduced target and pre-reduced generated cells must share one
    basis (the usual way: build both from the same preprocessing).
    """
    feat = x2d.shape[-1]
    in_dim = pca.components.shape[1]   # raw-feature (gene) dimension the PCA was fit on
    out_dim = pca.components.shape[0]  # number of components (reduced dimension)
    if feat == in_dim:
        return pca.transform(x2d)
    if feat == out_dim:
        return np.asarray(x2d)
    raise ValueError(
        f"Generated features (dim {feat}) match neither the target PCA's input dim {in_dim} "
        f"(raw features → would be projected) nor its output dim {out_dim} (already reduced). "
        "Make sure the generated cells share the target's feature space."
    )


@dataclass
class TargetSlide:
    """A real target slide: every cell's expression + coordinates, plus optional labels.

    Attributes:
        x: ``(N_cells, n_features)`` expression of all real cells (raw genes, or PCA if ``n_pcs``
            was requested when loading).
        pos: ``(N_cells, coord)`` spatial coordinates.
        ct: ``(N_cells,)`` integer cell-type labels (optional; used by the classifier metrics).
        n_classes: number of cell types (for proportion histograms).
        pca: the frozen PCA fit on this slide when ``n_pcs`` was given, else ``None``. Apply it to
            generated cells via :meth:`GeneratedNiches.project` so both sides share a basis.
    """

    x: np.ndarray
    pos: np.ndarray
    ct: np.ndarray | None = None
    n_classes: int | None = None
    pca: object | None = None
    # Optional per-slide coordinate standardiser (set on the shared-PCA path). Lets a caller map a
    # model's raw coordinates into this target's standardised frame before evaluating; ``None`` means
    # coordinates are already in the target's frame (nothing to do).
    coord_transform: object | None = None

    @property
    def moran_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """The (x, pos) used as the Moran's I real reference — the full target cloud.

        Moran's I now scores **all** generated cells against the full real slide, so there is no
        density-matched grid subsample any more; this returns the whole cloud.
        """
        return self.x, self.pos

    @classmethod
    def from_anndata(
        cls,
        adata_or_path,
        *,
        timepoint: str | None = None,
        expr_key: str | None = None,
        spatial_key: str = "spatial",
        ct_key: str | None = None,
        timepoint_key: str | None = None,
        n_pcs: int | None = None,
    ) -> TargetSlide:
        """Build a ``TargetSlide`` from an original AnnData slide (raw genes + coordinates).

        Args:
            adata_or_path: an ``AnnData`` or a path to a ``.h5ad`` file.
            timepoint: if the file holds several slides (``timepoint_key`` set), keep only this one.
            expr_key: where expression lives — ``None`` -> ``adata.X`` (raw genes); otherwise an
                ``obsm``/``layers`` key (e.g. ``"X_pca"`` if you already reduced).
            spatial_key: ``obsm`` key for coordinates (default ``"spatial"``).
            ct_key: ``obs`` column with cell types (optional).
            timepoint_key: ``obs`` column identifying the slide (optional).
            n_pcs: if set, fit a PCA on this slide and project its expression into ``n_pcs`` dims;
                the fit is stored on ``.pca`` so generated cells can be projected identically.
        """
        adata = read_anndata(adata_or_path)
        if timepoint is not None and timepoint_key is not None:
            mask = np.asarray(adata.obs[timepoint_key]).astype(str) == str(timepoint)
            adata = adata[mask]

        x = slide_expression(adata, expr_key)
        pos = slide_coords(adata, spatial_key)
        ct, ct_to_int = cell_type_labels(adata, ct_key)
        n_classes = len(ct_to_int) if ct_to_int is not None else None

        pca = None
        if n_pcs is not None and (expr_key is None or x.shape[1] > n_pcs):
            pca = fit_pca(x, n_pcs)
            x = pca.transform(x)

        return cls(x=x, pos=pos, ct=ct, n_classes=n_classes, pca=pca)

    @classmethod
    def from_dataclass(
        cls,
        ds_or_path,
        *,
        timepoint: str | None = None,
        shared_pca: bool = False,
        apply_lognorm: bool = True,
    ) -> TargetSlide:
        """Build a ``TargetSlide`` from a **preprocessed-slide pickle** (an ``H5ADDatasetDataclass``).

        For inputs already reduced to a shared PCA space: expression comes from ``X_pca``, coordinates
        from ``coords`` and labels from ``ct``. This is the schema some generators (e.g. the NicheFlow
        adapter) train and sample in; their output is then directly comparable.

        Args:
            ds_or_path: an ``H5ADDatasetDataclass`` or a path to a ``.pkl``.
            timepoint: which slide to use as the target (default: the last in ``timepoints_ordered``).
            shared_pca: when ``True``, reconstruct the dataclass's gene -> X_pca recipe as a
                :class:`~paired_slides_eval.data.shared_pca.SharedGenePCA` and attach it to ``.pca``,
                plus a :class:`~paired_slides_eval.data.shared_pca.CoordStandardizer` on
                ``.coord_transform``. This lets a model that emits **gene-space** cells + **raw**
                coordinates (the OT-CFM baseline) be projected into the *same* whitened-PCA +
                standardised-coord space the niche models live in — so all models share one basis.
                Cells already reduced to that PCA pass through unchanged (dimension auto-detect). The
                pickle must carry the recipe stats (``preprocess_pair`` persists them).
            apply_lognorm: forwarded to ``SharedGenePCA`` — set ``False`` if the gene-space cells are
                already log-normalised (see ``docs/comparability_plan.md``). Ignored unless
                ``shared_pca``.
        """
        if hasattr(ds_or_path, "X_pca"):
            ds = ds_or_path
        else:
            from paired_slides_eval.data.dataclass import load_h5ad_dataset_dataclass

            ds = load_h5ad_dataset_dataclass(ds_or_path)
        t = timepoint or ds.timepoints_ordered[-1]
        idx = np.asarray(ds.timepoint_indices[t])

        x = np.asarray(ds.X_pca)[idx]
        pos = np.asarray(ds.coords)[idx]
        ct_raw = np.asarray(ds.ct)[idx]
        if np.issubdtype(ct_raw.dtype, np.integer):
            ct = ct_raw.astype(np.int64)
        else:
            ct = np.array([ds.ct_to_int[c] for c in ct_raw], dtype=np.int64)
        n_classes = len(ds.ct_ordered)

        pca = coord_transform = None
        if shared_pca:
            from paired_slides_eval.data.shared_pca import (
                coord_standardizer_from_dataclass,
                shared_pca_from_dataclass,
            )

            pca = shared_pca_from_dataclass(ds, apply_lognorm=apply_lognorm)
            coord_transform = coord_standardizer_from_dataclass(ds, t)

        return cls(
            x=x, pos=pos, ct=ct, n_classes=n_classes, pca=pca, coord_transform=coord_transform
        )


@dataclass
class GeneratedNiches:
    """Generated microenvironments: ``(B, N, D)`` with the centroid at point index 0.

    Attributes:
        x: ``(B, N, n_features)`` generated expression.
        pos: ``(B, N, coord)`` generated coordinates.
        gt_x / gt_pos: the matched ground-truth (paired real) target microenvironment per
            generated niche; supply these to enable the pointwise regression metrics
            (``x/*``, ``pos/*``) and the cell-type classifier metrics (``ct/*``).
        gt_ct: ``(B,)`` true cell-type label of each paired real centroid; enables the classifier
            accuracy-gap metric (``ct/acc_real``, ``ct/acc_gen``, ``ct/acc_gap``).
    """

    x: np.ndarray
    pos: np.ndarray
    gt_x: np.ndarray | None = None
    gt_pos: np.ndarray | None = None
    gt_ct: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.pos = np.asarray(self.pos)
        if self.x.ndim != 3 or self.pos.ndim != 3:
            raise ValueError(
                f"GeneratedNiches expects (B, N, D) arrays; got x{self.x.shape} pos{self.pos.shape}"
            )

    def to_slide(self) -> GeneratedSlide:
        """Flatten the niches into a :class:`GeneratedSlide` — all niche points pooled.

        The single flattening path: every cell of every niche becomes one row in an
        ``(B*N, D)`` cloud. Overlapping niches therefore repeat shared cells (the points are the
        cells the niche model emitted). The label-free metrics consume this; ``flat_x`` /
        ``flat_pos`` are thin views over it.
        """
        return GeneratedSlide(
            x=self.x.reshape(-1, self.x.shape[-1]),
            pos=self.pos.reshape(-1, self.pos.shape[-1]),
        )

    @property
    def flat_x(self) -> np.ndarray:
        """All generated cells as one ``(B*N, n_features)`` cloud (see :meth:`to_slide`)."""
        return self.to_slide().x

    @property
    def flat_pos(self) -> np.ndarray:
        """All generated cells pooled into one ``(B*N, coord)`` cloud (via :meth:`to_slide`)."""
        return self.to_slide().pos

    @property
    def centroid_x(self) -> np.ndarray:
        """Centroid expression of each niche, ``(B, n_features)`` (point 0)."""
        return self.x[:, 0, :]

    @property
    def centroid_pos(self) -> np.ndarray:
        """Centroid coordinates of each niche, ``(B, coord)`` (point 0)."""
        return self.pos[:, 0, :]

    def project(self, pca) -> GeneratedNiches:
        """Project expression (and ``gt_x``) through a target ``pca`` into the shared basis.

        No-op when ``pca`` is ``None``. When ``pca`` is set, the space is auto-detected by feature
        dimension (see :func:`_pca_aware_transform`): gene-space cells are projected; cells already
        in the PCA basis are passed through unchanged. Coordinates are untouched.
        """
        if pca is None:
            return self

        def _proj(arr):
            if arr is None:
                return None
            b, n, _ = arr.shape
            return _pca_aware_transform(arr.reshape(-1, arr.shape[-1]), pca).reshape(b, n, -1)

        return GeneratedNiches(
            x=_proj(self.x), pos=self.pos, gt_x=_proj(self.gt_x), gt_pos=self.gt_pos,
            gt_ct=self.gt_ct,
        )

    @classmethod
    def from_anndata(
        cls,
        adata_or_path,
        *,
        niche_key: str = "niche_id",
        expr_key: str | None = None,
        spatial_key: str = "spatial",
        gt_x_key: str | None = "gt_x",
        gt_pos_key: str | None = "gt_pos",
        gt_ct_key: str | None = "gt_ct",
    ) -> GeneratedNiches:
        """Build ``(B, N, D)`` niches from a flat generated AnnData (the pipeline's output).

        The generated cells are stored flat (``B*N`` rows) with ``obs[niche_key]`` grouping the
        ``N`` points of each niche (centroid first) and coordinates in ``obsm[spatial_key]``. The
        paired ground-truth niches, if present, are carried in ``obsm[gt_x_key]`` /
        ``obsm[gt_pos_key]`` with the same row order; the paired real centroid's true label in
        ``obs[gt_ct_key]``.
        """
        adata = read_anndata(adata_or_path)
        if niche_key not in adata.obs:
            raise KeyError(
                f"niche_key {niche_key!r} not in adata.obs (have: {list(adata.obs.columns)})."
            )
        niche = np.asarray(adata.obs[niche_key])
        x = slide_expression(adata, expr_key)
        pos = slide_coords(adata, spatial_key)

        # Group rows by niche id, preserving within-niche order (centroid at index 0).
        order = np.argsort(niche, kind="stable")
        niche_sorted = niche[order]
        _, counts = np.unique(niche_sorted, return_counts=True)
        if len(set(counts.tolist())) != 1:
            raise ValueError(
                "Generated niches must all have the same number of points; "
                f"got sizes {set(counts)}."
            )
        n = int(counts[0])
        b = len(counts)

        def _reshape(flat):
            return flat[order].reshape(b, n, flat.shape[-1])

        gt_x = gt_pos = gt_ct = None
        if gt_x_key and gt_x_key in adata.obsm and gt_pos_key and gt_pos_key in adata.obsm:
            gt_x = _reshape(np.asarray(adata.obsm[gt_x_key], dtype=np.float32))
            gt_pos = _reshape(np.asarray(adata.obsm[gt_pos_key], dtype=np.float32))
        if gt_ct_key and gt_ct_key in adata.obs:
            # One label per niche: take the centroid (point 0) of each group.
            gt_ct = np.asarray(adata.obs[gt_ct_key])[order].reshape(b, n)[:, 0].astype(np.int64)

        return cls(x=_reshape(x), pos=_reshape(pos), gt_x=gt_x, gt_pos=gt_pos, gt_ct=gt_ct)

    @classmethod
    def from_trajectory(cls, x_traj, pos_traj, **kwargs) -> GeneratedNiches:
        """Build from a flow sampling trajectory: take the final step ``[-1]`` of each."""
        import numpy as _np

        def _last(t):
            arr = t[-1]
            return arr.detach().cpu().numpy() if hasattr(arr, "detach") else _np.asarray(arr)

        return cls(x=_last(x_traj), pos=_last(pos_traj), **kwargs)


@dataclass
class GeneratedSlide:
    """Generated cells as a **flat** slide: ``x (N, D)`` + ``pos (N, coord)``.

    For whole-slide generative models that emit a tissue directly, with no niche/microenvironment
    structure. The label-free metrics — ``psd``, ``spd``, ``distribution``, ``c2st``, ``moran`` —
    consume the flat cloud directly. The classifier metrics (``concordance``, ``ct_gap``) are
    computed by reconstructing microenvironments from geometry (each cell's neighbourhood paired to
    the nearest real cell's), so they run too; only ``regression`` needs a :class:`GeneratedNiches`
    with cell-for-cell matched ground truth and is skipped for a ``GeneratedSlide``.

    Exposes the same ``flat_x`` / ``flat_pos`` / ``project`` interface as
    :class:`GeneratedNiches`, so :func:`paired_slides_eval.evaluate.evaluate` accepts either.
    """

    x: np.ndarray  # (N, n_features)
    pos: np.ndarray  # (N, coord)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.pos = np.asarray(self.pos)
        if self.x.ndim != 2 or self.pos.ndim != 2:
            raise ValueError(
                f"GeneratedSlide expects (N, D) arrays; got x{self.x.shape} pos{self.pos.shape}. "
                "For niche-shaped (B, N, D) cells use GeneratedNiches."
            )

    @property
    def flat_x(self) -> np.ndarray:
        """All generated cells as an ``(N, n_features)`` cloud (already flat)."""
        return self.x

    @property
    def flat_pos(self) -> np.ndarray:
        """All generated cells as an ``(N, coord)`` cloud (already flat)."""
        return self.pos

    def to_slide(self) -> GeneratedSlide:
        """Return ``self`` — a flat slide is already flat (symmetry with ``GeneratedNiches``)."""
        return self

    def project(self, pca) -> GeneratedSlide:
        """Project expression into the target's basis, or pass through if already PCA-reduced.

        No-op when ``pca`` is ``None``; otherwise the space is auto-detected by feature dimension
        (see :func:`_pca_aware_transform`) — gene-space cells are projected, already-reduced cells
        are returned unchanged.
        """
        if pca is None:
            return self
        return GeneratedSlide(x=_pca_aware_transform(self.x, pca), pos=self.pos)

    @classmethod
    def from_anndata(
        cls,
        adata_or_path,
        *,
        expr_key: str | None = None,
        spatial_key: str = "spatial",
    ) -> GeneratedSlide:
        """Build a flat ``GeneratedSlide`` from a generated AnnData (one row per cell).

        Reads expression from ``adata.X`` (or ``expr_key``) and coordinates from
        ``obsm[spatial_key]`` — the same layout as :meth:`TargetSlide.from_anndata`, just for the
        generated cells. No ``niche_id`` grouping is needed.
        """
        adata = read_anndata(adata_or_path)
        return cls(x=slide_expression(adata, expr_key), pos=slide_coords(adata, spatial_key))
