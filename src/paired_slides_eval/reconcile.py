"""Coordinate reconciliation: map generated coordinates into the target's standardised frame.

Expression is reconciled by the caller (``GeneratedSlide.project(target.pca)``); this handles the
*coordinate* half centrally so every metric sees one consistent frame.
"""

from __future__ import annotations

import numpy as np

from paired_slides_eval.contract import GeneratedNiches, GeneratedSlide


def _standardize_generated_coords(generated, coord_transform):
    """Map a generated slide/niches' coordinates into the target's standardised frame.

    Puts the generated coords in the same per-slide standardised frame the niche models and the
    classifier live in, so ``moran``/``c2st`` and the niche pairing are on a common
    scale. No-op when ``coord_transform`` is ``None``. Only the *generated* coordinates are touched —
    paired ``gt_pos`` already comes from the target (already standardised).
    """
    if coord_transform is None:
        return generated
    if isinstance(generated, GeneratedSlide):
        return GeneratedSlide(x=generated.x, pos=coord_transform.transform(generated.pos))
    b, n, _ = generated.pos.shape
    pos = coord_transform.transform(generated.pos.reshape(-1, generated.pos.shape[-1]))
    return GeneratedNiches(
        x=generated.x, pos=pos.reshape(b, n, -1), gt_x=generated.gt_x,
        gt_pos=generated.gt_pos, gt_ct=generated.gt_ct,
    )


def _detect_coord_space(gen_pos: np.ndarray, coord_transform) -> str:
    """Detect whether ``gen_pos`` is in the target's RAW frame (-> standardise) or already standardised.

    Standardised coords have per-axis std ~1; raw coords have std ~ the target's raw coord std (stored
    on ``coord_transform``). Picks whichever the generated per-axis std is closer to (log-ratio), so it
    is robust whether the raw std is large (the usual case) or itself near 1 (then either choice is a
    no-op anyway). Returns ``"standardize"`` or ``"passthrough"``.
    """
    eps = 1e-8
    gen_std = np.asarray(gen_pos, dtype=np.float64).std(axis=0) + eps
    raw_std = np.asarray(coord_transform.std, dtype=np.float64) + eps
    to_standardised = np.abs(np.log(gen_std)).mean()       # distance to std == 1
    to_raw = np.abs(np.log(gen_std / raw_std)).mean()      # distance to the target's raw std
    return "passthrough" if to_standardised <= to_raw else "standardize"


def _reconcile_generated(generated, target, *, coords: str = "auto"):
    """Bring ``generated`` coordinates into the target's frame; return ``(generated, notes)``.

    Expression is already reconciled by the caller via ``.project(target.pca)`` (gene-space ->
    projected, already-reduced -> passthrough). This handles the *coordinate* half centrally so every
    metric sees a consistent frame:

    * ``"auto"`` (default) — if the target carries a standardised coord frame
      (``target.coord_transform``, set for shared-PCA pickles), detect whether the generated coords are
      raw or already standardised and reconcile accordingly, recording the decision in the notes. This
      removes the old silent-mismatch footgun (forgetting to standardise OT-CFM coords).
    * ``"standardize"`` / ``"passthrough"`` — force the choice.
    """
    if coords not in ("auto", "standardize", "passthrough"):
        raise ValueError(f"coords must be auto|standardize|passthrough, got {coords!r}")
    notes: list[str] = []
    ct = target.coord_transform
    if ct is None:
        if coords == "standardize":
            raise ValueError(
                "coords='standardize' needs a shared-PCA .pkl target (it carries the coord frame); "
                "the given target has none."
            )
        return generated, notes

    decision = coords
    if coords == "auto":
        decision = _detect_coord_space(generated.flat_pos, ct)
        gs = np.round(np.asarray(generated.flat_pos).std(axis=0), 2).tolist()
        rs = np.round(np.asarray(ct.std), 2).tolist()
        notes.append(
            f"coords auto -> {decision} (generated per-axis std {gs} vs target raw std {rs}; "
            "standardised coords have std ~1)"
        )
    if decision == "standardize":
        generated = _standardize_generated_coords(generated, ct)
    return generated, notes
