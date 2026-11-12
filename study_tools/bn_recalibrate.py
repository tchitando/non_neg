#!/usr/bin/env python3
"""Recompute the BatchNorm running statistics of a checkpoint whose weights are finite but whose BatchNorm running
statistics are not (a 16-bit overflow during pretraining poisons the running averages; the weights survive because
the step with non-finite gradients is skipped). Such runs are listed in study/nonfinite_runs.txt.

From the repository root on the server:

    CUDA_VISIBLE_DEVICES=1 python study_tools/bn_recalibrate.py --run ncl-resnet50-octmnist-croponly_pixelate-s43 --dry-run
    CUDA_VISIBLE_DEVICES=1 python study_tools/bn_recalibrate.py --run ncl-resnet50-octmnist-croponly_pixelate-s43

What it does:
  1. finds <ckpt-root>/**/<run>-offline-*-ep=99.ckpt and the args.json next to it (the run's own configuration);
  2. checks that every weight and bias is finite and that the only non-finite tensors are BatchNorm running
     statistics; otherwise it stops (that run can only be retrained);
  3. rebuilds the run's own training pipeline from args.json (same augmentation file, same number of views, same
     batch size, shuffled, drop_last, seeded with the run's seed) and passes one epoch of training views through the
     backbone in training mode, with no gradients and no weight changes. Every BatchNorm is reset first and then
     keeps a cumulative average of the batch statistics (momentum None), in float32;
  4. writes a copy of the checkpoint with the new statistics to <out-root>/<same relative path> and copies args.json;
     a bnfix.json next to it records what was done. The original checkpoint is not touched;
  5. checks that the new checkpoint gives finite h and z on validation images (clean, as eval_run.py reads them).

Score it afterwards with eval_run.py --ckpt-root trained_models_bnfix (the master table then records that path).
"""
import argparse, glob, hashlib, json, os, random, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")
import ctypes as _ctypes                                   # ImageMagick before torch, as in main_pretrain.py
for _lib in ("libMagickCore-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so.10"):
    _p = os.path.join(sys.prefix, "lib", _lib)
    if os.path.exists(_p):
        try:
            _ctypes.CDLL(_p, mode=_ctypes.RTLD_GLOBAL)
        except OSError:
            pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

STATS = ("running_mean", "running_var")


# ---------------------------------------------------------------- checkpoint
def find_ckpt(run_id, root):
    hits = sorted(glob.glob(os.path.join(root, "**", f"{run_id}-offline-*-ep=99.ckpt"), recursive=True), key=os.path.getmtime)
    if len(hits) != 1:
        raise SystemExit(f"expected one epoch-99 checkpoint for {run_id} under {root}, found {len(hits)}: {hits}")
    return hits[0]


def check_state(sd):
    """Split the float tensors into BatchNorm running statistics and everything else; report what is non-finite."""
    bad_stats, bad_other, n = [], [], 0
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        n += 1
        if not torch.isfinite(v.float()).all():
            (bad_stats if k.endswith(STATS) else bad_other).append(k)
    return n, bad_stats, bad_other


def build_models(sd):
    from torchvision import models
    bb = models.resnet50(weights=None)
    bb.fc = nn.Identity()
    pj = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256))
    bb.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    pj.load_state_dict({k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    return bb, pj


# ---------------------------------------------------------------- data
def build_loader(cfg, seed):
    """The run's own pretraining views, built exactly as MedMNIST/main_pretrain.py builds them."""
    from MedMNIST.pretrain_dataloader import (FullTransformPipeline, NCropAugmentation, build_transform_pipeline,
                                              prepare_dataloader, prepare_datasets)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    pipelines = [NCropAugmentation(build_transform_pipeline(cfg.data.dataset, a), a.num_crops,
                                   sharing=float(a.get("sharing", 0.0))) for a in cfg.augmentations]
    ds = prepare_datasets(cfg.data.dataset, FullTransformPipeline(pipelines), train_data_path=cfg.data.train_path,
                          data_format=cfg.data.format, no_labels=cfg.data.no_labels, data_fraction=cfg.data.fraction)
    return prepare_dataloader(ds, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers), pipelines


def views_of(batch):
    """A pretraining batch is (indexes, [view_1, ..., view_k], targets)."""
    return batch[1]


# ---------------------------------------------------------------- the recalibration
def reset_bn(bb):
    n = 0
    for m in bb.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                              # cumulative average over every batch seen
            n += 1
    return n


@torch.no_grad()
def recalibrate(bb, loader, device, max_batches=None, log_every=50):
    """Training-mode forward passes, one per view, as in pretraining (each view batch goes through the backbone on its
    own). No gradients, no optimiser: only the BatchNorm buffers change."""
    bb.train()
    nb, t0 = 0, time.time()
    for batch in loader:
        for x in views_of(batch):
            bb(x.to(device, non_blocking=True).float())
        nb += 1
        if log_every and nb % log_every == 0:
            print(f"  {nb} batches ({time.time() - t0:.0f}s)", flush=True)
        if max_batches and nb >= max_batches:
            break
    bb.eval()
    return nb


def new_state(sd, bb):
    """The original state dict with the backbone's BatchNorm buffers (statistics and batch counters) replaced."""
    out = dict(sd)
    for name, buf in bb.named_buffers():
        key = "backbone." + name
        if key not in sd:
            raise SystemExit(f"buffer {key} is not in the checkpoint")
        out[key] = buf.detach().cpu().to(sd[key].dtype)
    return out


@torch.no_grad()
def check_features(bb, pj, X, device, relu):
    bb.eval(); pj.eval()
    h = bb(X.to(device).float())
    z = pj(h)
    z = F.relu(z) if relu else z
    return bool(torch.isfinite(h).all() and torch.isfinite(z).all()), float(h.mean()), float(h.std()), float(z.abs().mean())


def val_images(n, seed=0):
    from torchvision import transforms
    from MedMNIST.octmnist_dataset import OCTMnistDataset, OCTMNIST_MEAN, OCTMNIST_STD
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(OCTMNIST_MEAN, OCTMNIST_STD)])
    ds = OCTMnistDataset(split="val", transform=tf)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    return torch.stack([ds[int(i)][0] for i in idx])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run id, e.g. ncl-resnet50-octmnist-croponly_pixelate-s43")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--out-root", default="trained_models_bnfix")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = one full epoch of training views")
    ap.add_argument("--check-images", type=int, default=1024, help="validation images for the final finite check")
    ap.add_argument("--dry-run", action="store_true", help="checks, the pipeline and one batch; nothing written")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output checkpoint")
    a = ap.parse_args()
    from omegaconf import OmegaConf

    src = find_ckpt(a.run, a.ckpt_root)
    args_json = Path(src).parent / "args.json"
    if not args_json.exists():
        raise SystemExit(f"{args_json} not found: the run's configuration is needed to rebuild its views")
    cfg = OmegaConf.create(json.loads(args_json.read_text()))
    if cfg.get("name") and cfg.name != a.run:
        raise SystemExit(f"args.json belongs to {cfg.name}, not {a.run}")
    print(f"checkpoint  {src}\nconfig      {args_json}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    n, bad_stats, bad_other = check_state(sd)
    print(f"tensors     {n} float tensors; non-finite: {len(bad_stats)} BatchNorm statistics, {len(bad_other)} others")
    if bad_other:
        raise SystemExit(f"weights or biases are non-finite ({bad_other[:4]} ...): BatchNorm recalibration cannot help; retrain")
    if not bad_stats:
        print("nothing to fix: every tensor is finite"); return

    relu = cfg.method_kwargs.get("non_neg", None) is not None
    print(f"run         {cfg.name} | dataset {cfg.data.dataset} | seed {cfg.seed} | batch {cfg.optimizer.batch_size} | "
          f"views {sum(int(x.num_crops) for x in cfg.augmentations)} | NCL read-out: {relu}")
    out = Path(a.out_root) / Path(src).relative_to(a.ckpt_root)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists; use --force to overwrite")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bb, pj = build_models(sd)
    bb, pj = bb.to(device), pj.to(device)
    loader, pipelines = build_loader(cfg, int(cfg.seed))
    print(f"pipeline    {pipelines}")
    print(f"loader      {len(loader)} batches per epoch")

    if a.dry_run:
        views = views_of(next(iter(loader)))
        print(f"one batch   {len(views)} views of shape {tuple(views[0].shape)}, dtype {views[0].dtype}, "
              f"pixel mean {float(views[0].mean()):.3f}")
        print(f"would write {out}\ndry run: nothing written"); return

    nbn = reset_bn(bb)
    t0 = time.time()
    nb = recalibrate(bb, loader, device, max_batches=a.max_batches or None)
    print(f"recalibrated {nbn} BatchNorm layers on {nb} batches ({time.time() - t0:.0f}s)")

    X = val_images(a.check_images)
    ok, hm, hs, zm = check_features(bb, pj, X, device, relu)
    print(f"check       {len(X)} validation images: finite {ok} | h mean {hm:.4f} SD {hs:.4f} | |z| mean {zm:.4f}")
    if not ok:
        raise SystemExit("features are still non-finite; nothing written")

    ck["state_dict"] = new_state(sd, bb)
    n2, bs2, bo2 = check_state(ck["state_dict"])
    if bs2 or bo2:
        raise SystemExit(f"new state still has non-finite tensors: {bs2[:3]} {bo2[:3]}; nothing written")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    (out.parent / "args.json").write_text(args_json.read_text())
    note = dict(run_id=a.run, source_checkpoint=str(src), output_checkpoint=str(out), date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                what="BatchNorm running statistics recomputed; every weight unchanged",
                how="reset, momentum None (cumulative average), training mode, no gradients, float32, the run's own training views",
                batches=nb, batch_size=int(cfg.optimizer.batch_size), views_per_image=sum(int(x.num_crops) for x in cfg.augmentations),
                seed=int(cfg.seed), nonfinite_before=len(bad_stats), batchnorm_layers=nbn,
                script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12])
    (out.parent / "bnfix.json").write_text(json.dumps(note, indent=1))
    print(f"wrote       {out}\n            {out.parent / 'bnfix.json'}")


if __name__ == "__main__":
    main()
