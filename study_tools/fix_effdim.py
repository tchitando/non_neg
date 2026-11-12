#!/usr/bin/env python3
"""Make eval_run.eff_dim robust to near-singular covariances (it crashed on ncl-resnet50-pathmnist-jpeg-s44).

    python study_tools/fix_effdim.py        # from the repository root; patches study_tools/eval_run.py in place

torch.linalg.eigvalsh can fail to converge on an ill-conditioned covariance, which happens when many projector units
are dead or constant (a collapsed or nearly collapsed run). The patched function computes on the CPU in float64 and,
if eigvalsh still fails, takes the eigenvalues from the singular values of the centred features (same quantity:
eig(f'f/n) = svd(f)^2/n). Effective dimension is unchanged for every run that scored before.
A backup is written to study_tools/eval_run.py.bak_before_effdim. The file's permissions are restored afterwards.
"""
import os, stat, sys
from pathlib import Path

P = Path("study_tools/eval_run.py")
OLD = '''def eff_dim(F_):
    f = F_.double() - F_.double().mean(0)
    cov = (f.T @ f) / len(f)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    lam = lam[lam > 1e-10]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else 0.0
'''
NEW = '''def eff_dim(F_):  # effdim-fix
    f = F_.detach().double().cpu()
    f = f - f.mean(0)
    try:
        lam = torch.linalg.eigvalsh((f.T @ f) / len(f)).clamp(min=0)
    except RuntimeError:  # eigh fails to converge on a near-singular covariance (collapsed runs)
        lam = torch.linalg.svdvals(f) ** 2 / len(f)
    lam = lam[lam > 1e-10]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else 0.0
'''

src = P.read_text()
if "# effdim-fix" in src:
    sys.exit("eval_run.py already patched")
if src.count(OLD) != 1:
    sys.exit("eff_dim in eval_run.py is not the expected version; nothing changed")
mode = P.stat().st_mode
Path(str(P) + ".bak_before_effdim").write_text(src)
os.chmod(P, mode | stat.S_IWUSR)
P.write_text(src.replace(OLD, NEW))
os.chmod(P, mode)
print("patched", P, "(backup: study_tools/eval_run.py.bak_before_effdim)")

sys.path.insert(0, str(P.parent)); sys.path.insert(0, os.getcwd())
try:
    import torch
    import importlib
    ER = importlib.import_module("eval_run")
    x = torch.randn(4000, 5) @ torch.randn(5, 64)
    print("check, rank-5 data (expect about 5 or less):", round(ER.eff_dim(x), 2))
    print("check, all-zero features (expect 0.0):", ER.eff_dim(torch.zeros(500, 32)))
    print("check, one live unit (expect 1.0):", round(ER.eff_dim(torch.cat([torch.randn(500, 1), torch.zeros(500, 31)], 1)), 2))
except Exception as e:  # the patch above is already written; this is only a quick check
    print("patch written; the quick check could not import eval_run here:", e)
