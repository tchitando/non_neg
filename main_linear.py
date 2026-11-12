# Copyright 2022 solo-learn development team.

import inspect
import logging
import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    from lightning.pytorch.loggers import WandbLogger

from pytorch_lightning.strategies import DDPStrategy
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from solo.args.linear import parse_cfg
from solo.data.classification_dataloader import prepare_data
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

try:
    from solo.data.dali_dataloader import ClassificationDALIDataModule
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]

    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    if cfg.backbone.name.startswith("resnet"):
        backbone.fc = nn.Identity()
        cifar = cfg.data.dataset in ["cifar10", "cifar100"]
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    ckpt_path = cfg.pretrained_feature_extractor
    assert ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".pth") or ckpt_path.endswith(".pt")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    backbone_state = {}
    projector_state = {}
    scale_param = 0

    for k, v in state_dict.items():
        if k.startswith("encoder."):
            k = k.replace("encoder.", "backbone.", 1)
            logging.warning("Older checkpoint format detected (encoder.*). Mapping to backbone.*")

        if k.startswith("backbone."):
            backbone_state[k.replace("backbone.", "", 1)] = v
        elif k.startswith("projector."):
            projector_state[k.replace("projector.", "", 1)] = v
        elif "scale_param" in k:
            scale_param = v

    msg_backbone = backbone.load_state_dict(backbone_state, strict=False)
    print("Backbone missing keys:", msg_backbone.missing_keys)
    print("Backbone unexpected keys:", msg_backbone.unexpected_keys)

    projector = nn.Sequential(
        nn.Linear(512, cfg.method_kwargs.proj_hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.method_kwargs.proj_hidden_dim, cfg.method_kwargs.proj_output_dim),
    )

    msg_projector = projector.load_state_dict(projector_state, strict=False)
    print("Projector missing keys:", msg_projector.missing_keys)
    print("Projector unexpected keys:", msg_projector.unexpected_keys)

    logging.info(f"Loaded {ckpt_path}")

    mixup_func = None
    mixup_active = cfg.mixup > 0 or cfg.cutmix > 0
    if mixup_active:
        logging.info("Mixup activated")
        mixup_func = Mixup(
            mixup_alpha=cfg.mixup,
            cutmix_alpha=cfg.cutmix,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=cfg.label_smoothing,
            num_classes=cfg.data.num_classes,
        )
        loss_func = SoftTargetCrossEntropy()
    elif cfg.label_smoothing > 0:
        loss_func = LabelSmoothingCrossEntropy(smoothing=cfg.label_smoothing)
    else:
        loss_func = torch.nn.CrossEntropyLoss()

    model = LinearModel(
        backbone,
        loss_func=loss_func,
        mixup_func=mixup_func,
        cfg=cfg,
        scale_param=scale_param,
    )

    for param in projector.parameters():
        param.requires_grad = False
    model.projector = projector.to("cuda")

    make_contiguous(model)

    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)

    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    train_loader, val_loader = prepare_data(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        auto_augment=cfg.auto_augment,
        noise_rate=cfg.noise_rate,
    )

    if cfg.data.format == "dali":
        assert _dali_avaliable, "Dali is not currently available, please install it first."
        assert not cfg.auto_augment, "Auto augmentation is not supported with Dali."

        dali_datamodule = ClassificationDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
        )
        dali_datamodule.val_dataloader = lambda: val_loader

    ckpt_path_resume, wandb_run_id = None, None
    if False and cfg.resume_from_checkpoint is None:
        auto_resumer = AutoResumer(
            checkpoint_dir=os.path.join(cfg.checkpoint.dir, "linear"),
            max_hours=cfg.auto_resume.max_hours,
        )
        resume_from_checkpoint, wandb_run_id = auto_resumer.find_checkpoint(cfg)
        if resume_from_checkpoint is not None:
            print("Resuming from previous checkpoint:", f"'{resume_from_checkpoint}'")
            ckpt_path_resume = resume_from_checkpoint
    elif cfg.resume_from_checkpoint is not None:
        ckpt_path_resume = cfg.resume_from_checkpoint
        del cfg.resume_from_checkpoint

    callbacks = []

    if cfg.checkpoint.enabled:
        ckpt_cb = Checkpointer(
            cfg,
            logdir=os.path.join(cfg.checkpoint.dir, "linear"),
            frequency=cfg.checkpoint.frequency,
            keep_prev=cfg.checkpoint.keep_prev,
        )
        callbacks.append(ckpt_cb)

    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline,
            resume="allow" if wandb_run_id else None,
            id=wandb_run_id,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))

        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
    else:
        wandb_logger = None

    trainer_kwargs = OmegaConf.to_container(cfg)
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {
        name: trainer_kwargs[name]
        for name in valid_kwargs
        if name in trainer_kwargs and trainer_kwargs[name] is not None
    }
    trainer_kwargs.update(
        {
            "logger": wandb_logger,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
        }
    )
    trainer = Trainer(**trainer_kwargs)

    try:
        from pytorch_lightning.loops import FitLoop

        class WorkaroundFitLoop(FitLoop):
            @property
            def prefetch_batches(self) -> int:
                return 1

        trainer.fit_loop = WorkaroundFitLoop(
            trainer.fit_loop.min_epochs, trainer.fit_loop.max_epochs
        )
    except Exception:
        pass

    if cfg.data.format == "dali":
        trainer.fit(model, ckpt_path=ckpt_path_resume, datamodule=dali_datamodule)
    else:
        trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path_resume)


if __name__ == "__main__":
    main()