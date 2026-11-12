import logging
from typing import Any, Callable, Dict, List, Tuple, Union

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau

from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from solo.utils.lars import LARS
from solo.utils.metrics import accuracy_at_k
from solo.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)


class LinearModel(pl.LightningModule):
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    _SCHEDULERS = ["reduce", "warmup_cosine", "step", "exponential", "none"]

    def __init__(
        self,
        backbone: nn.Module,
        cfg: omegaconf.DictConfig,
        loss_func: Callable = None,
        mixup_func: Callable = None,
        scale_param=0,
    ):
        super().__init__()

        cfg = self.add_and_assert_specific_cfg(cfg)

        self.backbone = backbone
        if hasattr(self.backbone, "inplanes"):
            features_dim = self.backbone.inplanes
        else:
            features_dim = self.backbone.num_features

        self.feature_selection = cfg.feature_selection
        if self.feature_selection is not None:
            features_dim = self.feature_selection

        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)

        self.mixup_func = mixup_func

        if loss_func is None:
            loss_func = nn.CrossEntropyLoss()
        self.loss_func = loss_func

        self.max_epochs = cfg.max_epochs
        self.accumulate_grad_batches = cfg.accumulate_grad_batches

        self.optimizer = cfg.optimizer.name
        self.batch_size = cfg.optimizer.batch_size
        self.lr = cfg.optimizer.lr
        self.weight_decay = cfg.optimizer.weight_decay
        self.extra_optimizer_args = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd = cfg.optimizer.exclude_bias_n_norm_wd
        self.layer_decay = cfg.optimizer.layer_decay

        self.scheduler = cfg.scheduler.name
        self.lr_decay_steps = cfg.scheduler.lr_decay_steps
        self.min_lr = cfg.scheduler.min_lr
        self.warmup_start_lr = cfg.scheduler.warmup_start_lr
        self.warmup_epochs = cfg.scheduler.warmup_epochs
        self.scheduler_interval = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]

        self.finetune = cfg.finetune
        self.no_channel_last = cfg.performance.disable_channel_last

        if not self.finetune:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(512, 16384),
            nn.ReLU(),
            nn.Linear(16384, 2048),
        )

        self.non_neg = cfg.non_neg
        self.SAE = cfg.SAE
        self.uniform_noise = cfg.uniform_noise
        self.gaussian_noise = cfg.gaussian_noise

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(
            cfg, "optimizer.exclude_bias_n_norm_wd", False
        )
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.0)

        cfg.finetune = omegaconf_select(cfg, "finetune", False)
        cfg.non_neg = omegaconf_select(cfg, "non_neg", None)
        cfg.SAE = omegaconf_select(cfg, "SAE", None)
        cfg.uniform_noise = omegaconf_select(cfg, "uniform_noise", 0.0)
        cfg.gaussian_noise = omegaconf_select(cfg, "gaussian_noise", 0.0)
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", None)

        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "step")

        cfg.performance = omegaconf_select(cfg, "performance", {})
        cfg.performance.disable_channel_last = omegaconf_select(
            cfg, "performance.disable_channel_last", False
        )

        cfg.feature_selection = omegaconf_select(cfg, "selected_dims", None)

        return cfg

    def configure_optimizers(self):
        if self.layer_decay > 0:
            assert self.finetune
            assert hasattr(self.backbone, "no_weight_decay")
            learnable_params = param_groups_layer_decay(
                self.backbone,
                self.weight_decay,
                no_weight_decay_list=self.backbone.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
            learnable_params.append({"name": "classifier", "params": self.classifier.parameters()})
        else:
            learnable_params = (
                self.classifier.parameters()
                if not self.finetune
                else [
                    {"name": "backbone", "params": self.backbone.parameters()},
                    {"name": "classifier", "params": self.classifier.parameters()},
                ]
            )

        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        optimizer_cls = self._OPTIMIZERS[self.optimizer]
        optimizer = optimizer_cls(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )

        if self.scheduler == "none":
            return optimizer

        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

        return [optimizer], [scheduler]

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)

        with torch.set_grad_enabled(self.finetune):
            feats = self.backbone(X)

            if self.feature_selection is not None:
                feats = self.projector(feats)
                if self.non_neg is not None:
                    feats = F.relu(feats)

            if not hasattr(self, "_printed_shape"):
                print("FEATURE SHAPE:", feats.shape)
                self._printed_shape = True

        logits = self.classifier(feats)
        return {"logits": logits, "input": X, "feats": feats.detach().cpu()}

    def shared_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        X, target = batch
        metrics = {"batch_size": X.size(0)}

        if self.training and self.mixup_func is not None:
            X, target = self.mixup_func(X, target)
            out = self(X)["logits"]
            loss = self.loss_func(out, target)
            metrics.update({"loss": loss})
        else:
            forward_out = self(X)
            out = forward_out["logits"]
            loss = self.loss_func(out, target)
            acc1, acc5 = accuracy_at_k(out, target, top_k=(1, 5))
            metrics.update(
                {
                    "loss": loss,
                    "acc1": acc1,
                    "acc5": acc5,
                    "targets": target,
                    "inputs": X,
                    "feats": forward_out["feats"],
                    "logits": out,
                }
            )

        return metrics

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if not self.finetune:
            self.backbone.eval()

        out = self.shared_step(batch, batch_idx)

        log = {"train_loss": out["loss"]}
        if self.mixup_func is None and "acc1" in out:
            log.update({"train_acc1": out["acc1"], "train_acc5": out["acc5"]})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        return out["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if self.uniform_noise > 0:
            sa, sb, sc, sd = batch[0].size()
            noise = torch.rand((sa, sb, sc, sd), device=batch[0].device)
            batch = (batch[0] + noise * self.uniform_noise, batch[1])

        if self.gaussian_noise > 0:
            sa, sb, sc, sd = batch[0].size()
            noise = torch.randn((sa, sb, sc, sd), device=batch[0].device)
            batch = (batch[0] + noise * self.gaussian_noise, batch[1])

        out = self.shared_step(batch, batch_idx)

        self.log("val_loss", out["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc1", out["acc1"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_acc5", out["acc5"], prog_bar=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        out = self.shared_step(batch, batch_idx)
        self.log("test_loss", out["loss"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_acc1", out["acc1"], prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("test_acc5", out["acc5"], prog_bar=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_epoch_end(self):
        pass