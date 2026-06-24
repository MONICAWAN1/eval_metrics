from collections.abc import Sequence
from typing import Any

import torch
import torchmetrics as tm
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from paired_slides_eval.classifier.dataset import CellTypeBatch
from paired_slides_eval.classifier.nets import CTClassifierNet
from paired_slides_eval.utils.log import RankedLogger
from paired_slides_eval.utils.plots import render_and_close

_logger = RankedLogger(__name__, rank_zero_only=True)


# Copied from https://github.com/martenlienen/unhippo/blob/main/unhippo/tasks/classification.py
@render_and_close
def plot_tm_metric(metric: tm.Metric):
    fig, _ = metric.plot()
    return fig


class Plots(Callback):
    def on_validation_epoch_end(self, trainer: Trainer, task: LightningModule) -> None:
        wandb_logger: WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.val_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="val_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_validation_epoch_end(trainer, task)

    def on_test_epoch_end(self, trainer: Trainer, task: LightningModule) -> None:
        wandb_logger: WandbLogger | None = trainer.logger
        if wandb_logger is None:
            return

        plots = [
            (metric_name, plot_tm_metric(metric))
            for metric_name, metric in task.test_plot_metrics.items()
        ]
        wandb_logger.log_image(
            key="test_plots",
            images=[img for _, img in plots],
            step=trainer.global_step,
            caption=[caption.removeprefix("Multiclass") for (caption, _) in plots],
        )
        return super().on_test_epoch_end(trainer, task)


class CellTypeClassification(LightningModule):
    def __init__(
        self,
        net: CTClassifierNet,
        optimizer: type[Optimizer],
        lr_scheduler: type[LRScheduler] | None = None,
        lr_scheduler_args: dict[str, Any] | None = None,
        plot_callbacks: bool = True,
        class_weight: Sequence[float] | str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "optimizer", "lr_scheduler"])

        self.net = net
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args
        self.plot_callbacks = plot_callbacks

        # Optional class weighting for the cross-entropy loss to counter class imbalance
        # (e.g. the GCN collapsing onto majority cell types). Accepts:
        #   None        -> unweighted (default)
        #   "balanced"  -> inverse-frequency weights computed from the train split at
        #                  fit start (sklearn-style: n_samples / (n_present * count_c),
        #                  absent classes get weight 0)
        #   list[float] -> explicit per-class weights (length == net.output_dim)
        self._class_weight = class_weight
        weight = None
        if isinstance(class_weight, (list, tuple)):
            weight = torch.as_tensor(list(class_weight), dtype=torch.float32)
        self.loss = nn.CrossEntropyLoss(weight=weight)

        # Effective microenvironment size the (spatial) classifier trains on, captured from
        # the datamodule at fit start and persisted in the checkpoint so eval can rebuild
        # identically sized niches. None for the gene-only classifier. See on_fit_start /
        # on_save_checkpoint.
        self._train_n_neighbors: int | None = None

        metrics = tm.MetricCollection(
            {
                "f1": tm.F1Score(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
                "auc": tm.AUROC(task="multiclass", num_classes=self.net.output_dim),
                "accuracy": tm.Accuracy(task="multiclass", num_classes=self.net.output_dim),
                "top3_acc": tm.Accuracy(
                    task="multiclass",
                    num_classes=self.net.output_dim,
                    top_k=3,
                    average="weighted",
                ),
                "precision": tm.Precision(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
                "recall": tm.Recall(
                    task="multiclass", num_classes=self.net.output_dim, average="weighted"
                ),
                # Macro (unweighted-by-support) variants surface minority-class
                # performance, which the weighted/accuracy metrics hide under class
                # imbalance. Use these to select a balanced model.
                "f1_macro": tm.F1Score(
                    task="multiclass", num_classes=self.net.output_dim, average="macro"
                ),
                "recall_macro": tm.Recall(
                    task="multiclass", num_classes=self.net.output_dim, average="macro"
                ),
            }
        )
        plot_metrics = tm.MetricCollection(
            {
                "ConfusionMatrix": tm.ConfusionMatrix(
                    task="multiclass", num_classes=self.net.output_dim
                ),
                "ROC": tm.ROC(task="multiclass", num_classes=self.net.output_dim),
                "PRCurve": tm.PrecisionRecallCurve(
                    task="multiclass", num_classes=self.net.output_dim
                ),
            }
        )
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.val_plot_metrics = plot_metrics.clone(prefix="val/")
        self.test_plot_metrics = plot_metrics.clone(prefix="test/")

    def on_fit_start(self) -> None:
        # Record the *effective* microenvironment size the spatial classifier trains on. The
        # dataset clamps the requested n_neighbors down to the smallest slide's neighbour count
        # (SpatialH5ADCTDataset), so this is the count the net actually sees — eval must rebuild
        # niches of the same size. None for the gene-only classifier (no n_neighbors).
        dataset = getattr(getattr(self.trainer, "datamodule", None), "dataset", None)
        n_neighbors = getattr(dataset, "n_neighbors", None)
        if n_neighbors is not None:
            self._train_n_neighbors = int(n_neighbors)

        # Compute data-driven class weights from the training split once the datamodule
        # has been set up and the module is on its device.
        #   "balanced"      -> inverse-frequency (1/count); strongest, but ultra-rare
        #                      classes get extreme weights
        #   "balanced_sqrt" -> 1/sqrt(count); tempers the rare-class extreme (recommended
        #                      under severe imbalance with near-singleton classes)
        if self._class_weight not in ("balanced", "balanced_sqrt") or self.loss.weight is not None:
            return
        weight = self._balanced_class_weights(sqrt=self._class_weight == "balanced_sqrt")
        self.loss = nn.CrossEntropyLoss(weight=weight.to(self.device))
        _logger.info(f"Using {self._class_weight} class weights: {weight.tolist()}")

    def _balanced_class_weights(self, sqrt: bool = False) -> Tensor:
        """Inverse-(sqrt-)frequency weights from the train split (absent classes -> 0)."""
        from torch.utils.data import Subset

        train = self.trainer.datamodule.train_dataset
        base = train.dataset if isinstance(train, Subset) else train
        indices = list(train.indices) if isinstance(train, Subset) else range(len(train))

        # Read labels cheaply without materialising the (expensive) neighbour gather.
        if hasattr(base, "ct_by_t") and hasattr(base, "index"):  # SpatialH5ADCTDataset
            labels = torch.tensor(
                [int(base.ct_by_t[base.index[i][0]][base.index[i][1]]) for i in indices]
            )
        elif hasattr(base, "ct"):  # H5ADCTDataset (gene-only)
            labels = base.ct[torch.as_tensor(list(indices))]
        else:  # generic fallback
            labels = torch.tensor([int(train[i]["y"]) for i in range(len(train))])

        n_classes = self.net.output_dim
        counts = torch.bincount(labels, minlength=n_classes).float()
        total = counts.sum()
        # Effective per-class count: count for inverse-freq, sqrt(count) for the gentler
        # scheme. Weights are normalised so the average per-cell weight is 1 (sklearn
        # "balanced" semantics): w_c = total / (eff_c * sum_c' eff_c' / total) ... i.e.
        # w_c = (total / sum eff) * (1 / eff_c) so that sum_c count_c*w_c == total.
        eff = counts.clamp_min(1.0).sqrt() if sqrt else counts.clamp_min(1.0)
        denom = (counts * (total / eff)).sum() / total  # = sum_c count_c / eff_c, scaled
        weight = (total / eff) / denom
        weight[counts == 0] = 0.0
        return weight

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Persist the effective microenvironment size at the checkpoint top level (not in the
        # net state_dict, so older checkpoints without it still load) so eval
        # (cell_type_concordance) can match the training niche size without hardcoding it.
        if self._train_n_neighbors is not None:
            checkpoint["n_neighbors"] = self._train_n_neighbors

    def training_step(self, batch: CellTypeBatch, _) -> dict[str, Tensor]:
        logits = self.net(batch["X"])
        loss = self.loss(logits, batch["y"])

        self.log("train/loss", loss.item(), batch_size=batch["X"].size(0), prog_bar=True)

        return {"loss": loss}

    def eval_step(
        self,
        batch: CellTypeBatch,
        metrics: tm.MetricCollection,
        plot_metrics: tm.MetricCollection,
    ) -> None:
        logits = self.net(batch["X"])
        self.log_dict(metrics(logits, batch["y"]))
        plot_metrics.update(logits, batch["y"])

    def validation_step(self, batch: CellTypeBatch, _) -> None:
        self.eval_step(batch, self.val_metrics, self.val_plot_metrics)

    def test_step(self, batch: CellTypeBatch, _) -> None:
        self.eval_step(batch, self.test_metrics, self.test_plot_metrics)

    def configure_optimizers(self) -> Any:
        optimizer = self._optimizer(
            params=self.net.parameters(),
        )
        config = {"optimizer": optimizer}
        if self._lr_scheduler is not None and self._lr_scheduler_args is not None:
            config["lr_scheduler"] = {
                "scheduler": self._lr_scheduler(optimizer=optimizer),
                **self._lr_scheduler_args,
            }
        return config

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        callbacks = super().configure_callbacks()
        if self.plot_callbacks:
            callbacks = [*callbacks, Plots()]
        return callbacks
