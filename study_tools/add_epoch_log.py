#!/usr/bin/env python3
"""Add a per-epoch log to MedMNIST/main_pretrain.py: after every training epoch, one row per metric goes to
<checkpoint dir>/metrics/<run name>.csv (trained_models/metrics/<run id>.csv), with columns epoch, global_step,
metric, value. Nothing about training, checkpoints or their names changes, and a failure inside the logger is printed
and ignored, so it can never stop a run.

Rows per epoch: everything the method already logs (train_nce_loss, train_class_loss, train_acc1/5 and NCL's
num_active_dim, non_neg_ratio, sparse_vals_ratio, effective_rank, orthogonality, semantic_consistency), the learning
rate, and two counts read from the model: bn_stats_nonfinite (backbone BatchNorm running statistics with NaN/inf) and
params_nonfinite (parameters with NaN/inf). A run that breaks shows the epoch it broke in.

From the repository root:
    python study_tools/add_epoch_log.py                          # writes MedMNIST/main_pretrain_log.py (a copy)
    python study_tools/add_epoch_log.py --out MedMNIST/main_pretrain.py --backup MedMNIST/main_pretrain_before_log.py
The first form leaves main_pretrain.py alone, so the copy can be tested on a short run before it replaces the original.
"""
import argparse, re, shutil
from pathlib import Path

CLASS = '''
# ---- per-epoch log (study_tools/add_epoch_log.py) -------------------------------------------------------------------
from pytorch_lightning import Callback as _EpochLogBase


class EpochMetricsCSV(_EpochLogBase):
    """After every training epoch: one row per metric -> <checkpoint dir>/metrics/<run name>.csv. Never stops training."""

    def __init__(self, cfg):
        self.path = None
        try:
            from pathlib import Path as _P
            d = _P(str(cfg.checkpoint.dir)) / "metrics"
            d.mkdir(parents=True, exist_ok=True)
            p, k = d / f"{cfg.name}.csv", 2
            while p.exists():                               # a rerun of the same run id gets its own file
                p, k = d / f"{cfg.name}_{k}.csv", k + 1
            with open(p, "w") as f:
                f.write("epoch,global_step,metric,value\\n")
            self.path = p
            print(f"[epoch log] {p}", flush=True)
        except Exception as e:
            print(f"[epoch log] off: {e}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.path is None:
            return
        try:
            import torch as _t
            rows = []
            for k, v in trainer.callback_metrics.items():
                try:
                    rows.append((k, float(v)))
                except Exception:
                    pass
            try:
                rows.append(("lr", float(trainer.optimizers[0].param_groups[0]["lr"])))
            except Exception:
                pass
            bb = getattr(pl_module, "backbone", None)
            if bb is not None:
                rows.append(("bn_stats_nonfinite", sum(int(not _t.isfinite(b).all()) for n, b in bb.named_buffers()
                                                       if n.endswith(("running_mean", "running_var")))))
            rows.append(("params_nonfinite", sum(int(not _t.isfinite(p).all()) for p in pl_module.parameters())))
            with open(self.path, "a") as f:
                for k, v in rows:
                    f.write(f"{trainer.current_epoch},{trainer.global_step},{k},{v}\\n")
        except Exception as e:
            print(f"[epoch log] epoch {getattr(trainer, 'current_epoch', '?')}: {e}", flush=True)
# ---------------------------------------------------------------------------------------------------------------------


'''
HOOK = "    callbacks.append(EpochMetricsCSV(cfg))\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="MedMNIST/main_pretrain.py")
    ap.add_argument("--out", default="MedMNIST/main_pretrain_log.py")
    ap.add_argument("--backup", default=None, help="copy the original here first (needed when --out is the original)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    text = src.read_text()
    if "EpochMetricsCSV" in text:
        raise SystemExit(f"{src} already has the epoch log")
    if len(re.findall(r"(?m)^@hydra\.main\(", text)) != 1:
        raise SystemExit("expected exactly one '@hydra.main(' line")
    if text.count("    callbacks = []\n") != 1:
        raise SystemExit("expected exactly one '    callbacks = []' line")
    new = re.sub(r"(?m)^(@hydra\.main\()", lambda m: CLASS.lstrip("\n") + m.group(1), text, count=1)
    new = new.replace("    callbacks = []\n", "    callbacks = []\n" + HOOK, 1)
    compile(new, str(out), "exec")                         # syntax check before anything is written
    if out.resolve() == src.resolve():
        if not a.backup:
            raise SystemExit("writing over the original needs --backup")
        if Path(a.backup).exists():
            raise SystemExit(f"{a.backup} exists; not overwriting it")
        shutil.copy2(src, a.backup)
        print(f"backup {a.backup}")
    elif out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    out.write_text(new)
    print(f"wrote {out} (compiles); added EpochMetricsCSV before @hydra.main and one callbacks.append line")


if __name__ == "__main__":
    main()
