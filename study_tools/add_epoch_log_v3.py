#!/usr/bin/env python3
"""Per-epoch log and NaN watch for MedMNIST/main_pretrain.py (version 3 of add_epoch_log.py: version 2 plus a per-step
diagnostic mode, off unless EPOCHLOG_DIAG=1, and NaN-aware size readouts).

Installs, or upgrades in place, a callback that:
  1. after every training epoch writes one row per metric to <checkpoint dir>/metrics/<run name>.csv
     (columns epoch, global_step, metric, value): everything the method logs (train_nce_loss, NCL's num_active_dim, ...),
     the learning rate, the NaN counts, and a size readout per stage (stem, layer1-4) of the backbone:
        bn_in_sd_max_<stage>      largest sqrt(running_var) of the BatchNorms in the stage: the spread of the
                                  activations going into BatchNorm (16-bit overflows above 65504)
        bn_in_absmean_max_<stage> largest |running_mean| in the stage
        bn_gamma_max_<stage>      largest |BatchNorm weight| in the stage
        conv_wnorm_<stage>        norm of all convolution weights in the stage
  2. every NAN_WATCH_EVERY training steps (default 20) checks the backbone's BatchNorm running statistics and every
     parameter. At the first NaN/inf it writes rows nan_detected (= the step), the counts and the size readout, copies
     the last saved (healthy) checkpoint to <its name>-last-finite.ckpt (that name never matches *-ep=99.ckpt, so no
     scoring script and no launcher picks it up), and stops the run. The launcher then finds no epoch-99 checkpoint and
     lists the run as NOT FINISHED, as it does for any broken run.
A failure inside the callback is printed and ignored. Training itself is unchanged.
Test hook: environment variable EPOCHLOG_TEST_NAN_AT_STEP=<step> puts a NaN into one BatchNorm buffer at that step.
Diagnostic mode (EPOCHLOG_DIAG=1): at every check (NAN_WATCH_EVERY, set it to 1 for every step) it also writes the step's
loss, the NaN counts, the size readout, and bn_in_absmax_<stage>: the largest |activation| entering any BatchNorm of the
stage during the steps since the last check (from forward pre-hooks; 16-bit overflow shows as inf).
EPOCHLOG_STOP_AT_STEP=<step> stops the run at that step (the learning-rate schedule is untouched, unlike max_steps).

From the repository root:
    python study_tools/add_epoch_log_v3.py --out MedMNIST/main_pretrain_diag.py          # a separate copy for diagnostics
The source may be the original main_pretrain.py or one with version 1 or 2; that block is replaced in the output.
"""
import argparse, re, shutil
from pathlib import Path

START = "# ---- per-epoch log (study_tools/add_epoch_log.py) "
CLASS = '''# ---- per-epoch log (study_tools/add_epoch_log.py) v3: epoch rows, size readout, NaN watch, diagnostics ---------------
from pytorch_lightning import Callback as _EpochLogBase


def _is_num(v):
    try:
        float(v)
        return True
    except Exception:
        return False


class EpochMetricsCSV(_EpochLogBase):
    """Epoch rows -> <checkpoint dir>/metrics/<run name>.csv; stops the run at the first NaN/inf. Never raises."""

    def __init__(self, cfg):
        import os as _os
        self.path, self.stopped, self.injected = None, False, False
        self.diag, self.stop_at, self.amax, self.hooked = False, None, {}, False
        try:
            self.diag = _os.environ.get("EPOCHLOG_DIAG", "") == "1"
            self.stop_at = int(_os.environ["EPOCHLOG_STOP_AT_STEP"]) if _os.environ.get("EPOCHLOG_STOP_AT_STEP") else None
            self.every = max(1, int(_os.environ.get("NAN_WATCH_EVERY", "20")))
            self.inject_at = int(_os.environ["EPOCHLOG_TEST_NAN_AT_STEP"]) if _os.environ.get("EPOCHLOG_TEST_NAN_AT_STEP") else None
            from pathlib import Path as _P
            d = _P(str(cfg.checkpoint.dir)) / "metrics"
            d.mkdir(parents=True, exist_ok=True)
            p, k = d / f"{cfg.name}.csv", 2
            while p.exists():                               # a rerun of the same run id gets its own file
                p, k = d / f"{cfg.name}_{k}.csv", k + 1
            with open(p, "w") as f:
                f.write("epoch,global_step,metric,value\\n")
            self.path = p
            print(f"[epoch log] {p} (NaN watch every {self.every} steps; diagnostics {self.diag}; stop at {self.stop_at})", flush=True)
        except Exception as e:
            print(f"[epoch log] off: {e}", flush=True)

    @staticmethod
    def _stage(name):
        return name.split(".")[0] if name.startswith("layer") else "stem"

    def _counts(self, pl_module):
        import torch as _t
        bb = getattr(pl_module, "backbone", None)
        bn = 0 if bb is None else sum(int(not _t.isfinite(b).all()) for n, b in bb.named_buffers()
                                      if n.endswith(("running_mean", "running_var")))
        pa = sum(int(not _t.isfinite(p).all()) for p in pl_module.parameters())
        return bn, pa

    def _sizes(self, pl_module):
        import torch as _t
        rows, bb = [], getattr(pl_module, "backbone", None)
        if bb is None:
            return rows
        sd, am, gm, wn = {}, {}, {}, {}
        for name, m in bb.named_modules():
            s = self._stage(name)
            if isinstance(m, _t.nn.modules.batchnorm._BatchNorm) and m.running_var is not None:
                v = float(m.running_var.detach().float().max().clamp_min(0).sqrt())
                u = float(m.running_mean.detach().float().abs().max())
                sd[s] = v if v != v or sd.get(s, 0.0) != sd.get(s, 0.0) else max(sd.get(s, 0.0), v)   # NaN wins
                am[s] = u if u != u or am.get(s, 0.0) != am.get(s, 0.0) else max(am.get(s, 0.0), u)
                if m.weight is not None:
                    gm[s] = max(gm.get(s, 0.0), float(m.weight.detach().float().abs().max()))
            elif isinstance(m, _t.nn.Conv2d):
                wn[s] = wn.get(s, 0.0) + float(m.weight.detach().float().pow(2).sum())
        for s in sorted(set(sd) | set(wn)):
            if s in sd:
                rows += [(f"bn_in_sd_max_{s}", sd[s]), (f"bn_in_absmean_max_{s}", am[s])]
            if s in gm:
                rows.append((f"bn_gamma_max_{s}", gm[s]))
            if s in wn:
                rows.append((f"conv_wnorm_{s}", wn[s] ** 0.5))
        return rows

    def _write(self, trainer, rows):
        with open(self.path, "a") as f:
            for k, v in rows:
                f.write(f"{trainer.current_epoch},{trainer.global_step},{k},{v}\\n")

    def _keep_last_healthy(self, trainer):
        import shutil as _sh, os as _os
        for cb in getattr(trainer, "callbacks", []):
            last = getattr(cb, "last_ckpt", None)
            if last and _os.path.exists(str(last)):
                dst = str(last)[:-len(".ckpt")] + "-last-finite.ckpt"
                _sh.copy2(str(last), dst)
                return dst
        return None

    def _hook(self, pl_module):
        import torch as _t
        bb = getattr(pl_module, "backbone", None)
        if bb is None:
            return
        for name, m in bb.named_modules():
            if isinstance(m, _t.nn.modules.batchnorm._BatchNorm):
                s = self._stage(name)
                def pre(mod, args, s=s):
                    x = args[0].detach().abs().amax().float()
                    cur = self.amax.get(s)
                    self.amax[s] = x if cur is None else _t.maximum(cur, x)
                m.register_forward_pre_hook(pre)
        self.hooked = True

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if self.path is None or self.stopped:
            return
        try:
            step = trainer.global_step
            if self.diag and not self.hooked:
                self._hook(pl_module)
            if self.stop_at is not None and step >= self.stop_at:
                self._write(trainer, [("stopped_at_step", step)])
                self.stopped = True
                trainer.should_stop = True
                print(f"[epoch log] stopping at step {step} (EPOCHLOG_STOP_AT_STEP)", flush=True)
                return
            if self.inject_at is not None and not self.injected and step >= self.inject_at:
                import torch as _t
                for name, m in pl_module.backbone.named_modules():
                    if name.startswith("layer1") and isinstance(m, _t.nn.modules.batchnorm._BatchNorm):
                        m.running_var.fill_(float("nan")); self.injected = True
                        print(f"[epoch log] TEST: NaN put into backbone.{name}.running_var at step {step}", flush=True)
                        break
            if step % self.every:
                return
            bn, pa = self._counts(pl_module)
            if self.diag:
                rows = [(k, float(v)) for k, v in trainer.callback_metrics.items() if "nce_loss" in k and _is_num(v)]
                rows += [("bn_stats_nonfinite", bn), ("params_nonfinite", pa)] + self._sizes(pl_module)
                rows += [(f"bn_in_absmax_{s}", float(v)) for s, v in sorted(self.amax.items())]
                self.amax = {}
                self._write(trainer, rows)
            if bn or pa:
                kept = self._keep_last_healthy(trainer)
                self._write(trainer, [("nan_detected", step), ("bn_stats_nonfinite", bn), ("params_nonfinite", pa)]
                            + self._sizes(pl_module))
                self.stopped = True
                trainer.should_stop = True
                print(f"[epoch log] NaN/inf at step {step} (epoch {trainer.current_epoch}): {bn} BatchNorm statistics, "
                      f"{pa} parameters. Stopping. Last healthy checkpoint kept: {kept}", flush=True)
        except Exception as e:
            print(f"[epoch log] step check: {e}", flush=True)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.path is None:
            return
        try:
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
            bn, pa = self._counts(pl_module)
            rows += [("bn_stats_nonfinite", bn), ("params_nonfinite", pa)] + self._sizes(pl_module)
            if (bn or pa) and not self.stopped:               # broke after the last step check: stop here
                kept = self._keep_last_healthy(trainer)
                rows.append(("nan_detected", trainer.global_step))
                self.stopped = True
                trainer.should_stop = True
                print(f"[epoch log] NaN/inf at the end of epoch {trainer.current_epoch}. Stopping. "
                      f"Last healthy checkpoint kept: {kept}", flush=True)
            self._write(trainer, rows)
        except Exception as e:
            print(f"[epoch log] epoch {getattr(trainer, 'current_epoch', '?')}: {e}", flush=True)
# ---------------------------------------------------------------------------------------------------------------------


'''
HOOK = "    callbacks.append(EpochMetricsCSV(cfg))\n"
BLOCK = re.compile(r"(?s)# ---- per-epoch log \(study_tools/add_epoch_log\.py\).*?\n# -{40,}\n\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="MedMNIST/main_pretrain.py")
    ap.add_argument("--out", required=True, help="e.g. MedMNIST/main_pretrain_log2.py, or the original with --backup")
    ap.add_argument("--backup", default=None, help="copy the original here first (needed when --out is the source)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    text = src.read_text()
    if START in text:                                       # version 1 (or 2) present: replace the block
        if len(BLOCK.findall(text)) != 1:
            raise SystemExit("found the epoch-log marker but not exactly one complete block; not touching it")
        if text.count(HOOK) != 1:
            raise SystemExit("expected exactly one 'callbacks.append(EpochMetricsCSV(cfg))' line")
        new = BLOCK.sub(lambda m: CLASS, text, count=1)
        what = "replaced the existing epoch-log block with version 2"
    else:                                                   # fresh install
        if len(re.findall(r"(?m)^@hydra\.main\(", text)) != 1:
            raise SystemExit("expected exactly one '@hydra.main(' line")
        if text.count("    callbacks = []\n") != 1:
            raise SystemExit("expected exactly one '    callbacks = []' line")
        new = re.sub(r"(?m)^(@hydra\.main\()", lambda m: CLASS + m.group(1), text, count=1)
        new = new.replace("    callbacks = []\n", "    callbacks = []\n" + HOOK, 1)
        what = "added the epoch-log block (version 2) and one callbacks.append line"
    if new.count("class EpochMetricsCSV") != 1:
        raise SystemExit("the result would not have exactly one EpochMetricsCSV; not writing")
    compile(new, str(out), "exec")                          # syntax check before anything is written
    if out.resolve() == src.resolve():
        if not a.backup:
            raise SystemExit("writing over the source needs --backup")
        if Path(a.backup).exists():
            raise SystemExit(f"{a.backup} exists; not overwriting it")
        shutil.copy2(src, a.backup)
        print(f"backup {a.backup}")
    elif out.exists():
        raise SystemExit(f"{out} exists; not overwriting it")
    out.write_text(new)
    print(f"wrote {out} (compiles); {what}")


if __name__ == "__main__":
    main()
