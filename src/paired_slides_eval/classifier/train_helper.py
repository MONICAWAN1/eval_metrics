"""Train a spatial cell-type classifier on a preprocessed slide, programmatically.

A thin wrapper over the local classifier stack (``SpatialH5ADCTDataset`` +
``CellTypeClassification`` + a Lightning ``Trainer``) so the pipeline can train the neutral
classifier from the classifier-slide dataclass without the Hydra CLI. The full Hydra training
entry point (``python -m paired_slides_eval.classifier.train``) still exists for configured runs.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path


def train_spatial_classifier(
    clf_dataclass,
    *,
    n_pcs: int,
    n_classes: int,
    n_neighbors: int = 32,
    hidden_dim: int = 64,
    num_heads: int = 4,
    coord_dim: int = 2,
    mask_centroid: bool = True,
    max_epochs: int = 20,
    batch_size: int = 1024,
    accelerator: str = "auto",
    seed: int = 2025,
):
    """Train a ``SpatialCTClassifierNet`` on ``clf_dataclass`` and return the frozen net.

    The returned net carries ``.n_neighbors`` (the effective microenvironment size) so the
    classifier metrics rebuild identically sized niches.
    """
    import lightning as L
    import torch

    from paired_slides_eval.classifier.datamodule import H5ADCTDataModule
    from paired_slides_eval.classifier.nets import SpatialCTClassifierNet
    from paired_slides_eval.classifier.task import CellTypeClassification

    with tempfile.TemporaryDirectory() as tmp:
        pkl_path = Path(tmp) / "clf.pkl"
        with pkl_path.open("wb") as fh:
            pickle.dump(clf_dataclass, fh, protocol=pickle.HIGHEST_PROTOCOL)

        dm = H5ADCTDataModule(
            data_fp=str(pkl_path),
            split_seed=seed,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=0,
            n_neighbors=n_neighbors,
        )
        dm.prepare_data()

        net = SpatialCTClassifierNet(
            input_dim=n_pcs,
            output_dim=n_classes,
            hidden_dim=hidden_dim,
            coord_dim=coord_dim,
            num_heads=num_heads,
            mask_centroid=mask_centroid,
        )
        task = CellTypeClassification(net=net, optimizer=torch.optim.Adam, plot_callbacks=False)

        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )
        trainer.fit(task, datamodule=dm)

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        # Effective niche size the net trained on (clamped in the dataset) so the metrics match.
        net.n_neighbors = int(getattr(dm.dataset, "n_neighbors", n_neighbors))
        return net
