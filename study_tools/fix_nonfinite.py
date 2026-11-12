#!/usr/bin/env python3
"""Make eval_run.py and eval_test.py skip a run whose features are not finite (a run that diverged to NaN/inf).

    python study_tools/fix_nonfinite.py            # from the repository root; patches both scripts in place
    python study_tools/fix_nonfinite.py --scan     # also list the PathMNIST checkpoints whose weights are not finite

Why: PathMNIST scoring stopped at one run whose backbone/projector output contains NaN or inf. eval_run.py crashed in
eff_dim (svdvals on non-finite input) and eval_test.py in sklearn ("Input contains NaN"); every run after it in the
queue was left unscored. With the patch, such a run is not scored: it is printed as SKIPPED and written to
nonfinite_runs.txt next to the results table (run id, checkpoint, which features), and the loop goes on. Nothing is
appended to the results tables for it, so a diverged run never enters an average. Finite runs score exactly as before.
Backups: study_tools/eval_run.py.bak_before_nonfinite and study_tools/eval_test.py.bak_before_nonfinite.
"""
import argparse, glob, os, stat, sys
from pathlib import Path

PATCHES = {
    "study_tools/eval_run.py": (
        '        hva, zva, yva = extract(bb, pj, "val", relu, a.bs, a.workers)\n',
        '        hva, zva, yva = extract(bb, pj, "val", relu, a.bs, a.workers)\n'
        '        bad = [n for n, t in (("h train", htr), ("z train", ztr), ("h val", hva), ("z val", zva))\n'
        '               if not torch.isfinite(t).all()]\n'
        '        if bad:  # nonfinite-guard: a diverged run is listed, not scored\n'
        '            nf = Path(a.master).parent / "nonfinite_runs.txt"\n'
        '            with open(nf, "a") as f:\n'
        '                f.write(f"{r[\'run_id\']}\\t{ckpt}\\tnon-finite: {\', \'.join(bad)}\\n")\n'
        '            print(f"    SKIPPED: non-finite features ({\', \'.join(bad)}); the run diverged. Listed in {nf}", flush=True)\n'
        '            del bb, pj, htr, ztr, hva, zva\n'
        '            torch.cuda.empty_cache()\n'
        '            continue\n',
    ),
    "study_tools/eval_test.py": (
        '        hte, zte_raw, yte = ER.extract(bb, pj, "test", False, a.bs, a.workers)\n',
        '        hte, zte_raw, yte = ER.extract(bb, pj, "test", False, a.bs, a.workers)\n'
        '        bad = [n for n, t in (("h train", htr), ("z train", ztr_raw), ("h test", hte), ("z test", zte_raw))\n'
        '               if not torch.isfinite(t).all()]\n'
        '        if bad:  # nonfinite-guard: a diverged run is listed, not scored\n'
        '            nf = out.parent / "nonfinite_runs_test.txt"\n'
        '            with open(nf, "a") as f:\n'
        '                f.write(f"{r[\'run_id\']}\\t{r[\'_ckpt\']}\\tnon-finite: {\', \'.join(bad)}\\n")\n'
        '            print(f"    SKIPPED: non-finite features ({\', \'.join(bad)}); the run diverged. Listed in {nf}", flush=True)\n'
        '            del bb, pj, htr, ztr_raw, hte, zte_raw\n'
        '            torch.cuda.empty_cache()\n'
        '            continue\n',
    ),
}


def patch(path, old, new):
    P = Path(path)
    if not P.exists():
        sys.exit(f"{path} not found: run this from the repository root")
    src = P.read_text()
    if "# nonfinite-guard" in src:
        print(f"{path}: already patched"); return
    if src.count(old) != 1:
        sys.exit(f"{path}: the anchor line was found {src.count(old)} times (expected 1); nothing changed. Send me the file.")
    mode = P.stat().st_mode
    Path(path + ".bak_before_nonfinite").write_text(src)
    os.chmod(P, mode | stat.S_IWUSR)
    P.write_text(src.replace(old, new))
    os.chmod(P, mode)
    compile(P.read_text(), path, "exec")
    print(f"{path}: patched (backup {path}.bak_before_nonfinite)")


def scan(root, pattern):
    import torch
    hits = sorted(glob.glob(os.path.join(root, "**", f"{pattern}-ep=99.ckpt"), recursive=True))
    print(f"\nscanning {len(hits)} epoch-99 checkpoints for non-finite weights (backbone + projector)")
    n_bad = 0
    for h in hits:
        sd = torch.load(h, map_location="cpu")["state_dict"]
        bad = [k for k, v in sd.items() if (k.startswith("backbone.") or k.startswith("projector."))
               and torch.is_floating_point(v) and not torch.isfinite(v).all()]
        if bad:
            n_bad += 1
            print(f"  NON-FINITE  {os.path.basename(h)}  ({len(bad)} tensors, e.g. {bad[0]})")
    print(f"{n_bad} checkpoint(s) with non-finite weights" + ("" if n_bad else " (so any NaN comes from activations, not weights)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--pattern", default="*pathmnist*", help="file-name pattern for --scan (default: PathMNIST runs)")
    a = ap.parse_args()
    for p, (old, new) in PATCHES.items():
        patch(p, old, new)
    if a.scan:
        scan(a.ckpt_root, a.pattern)
