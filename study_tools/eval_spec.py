#!/usr/bin/env python3
"""Specialisation audit: controls and a sign-agnostic specialisation measure for every scored run.

Run from the repository root on the server, after eval_run.py has scored the runs:

    python study_tools/eval_spec.py                       # every run in results_master.csv not yet audited
    python study_tools/eval_spec.py --random-init 3       # also 3 untrained networks per objective (controls)
    python study_tools/eval_spec.py --only '^ncl-'        # a subset
    python study_tools/eval_spec.py --selftest            # metric checks on synthetic data, no GPU or data needed

Reads checkpoint paths from study/results_master.csv (latest row per run_id) and appends one row per run to
study/results_spec.csv (append-only, like the master table). Only the VALIDATION split is read.

s2 changes (20 Oct 2026 review): z is read out as in training (SimCLR: l2-normalised; NCL: normalise -> ReLU ->
normalise), and MS uses a fixed class-balanced subset of 500 validation images per class (seed 0) instead of the
first 2000 images. s1 rows stay in the table; the latest row per run_id (s2) is the one used downstream.

Why each measure (all on a class-balanced validation subset, n images per class = smallest class, 5 draws):
    sc_star, sc_star_shuf   SC* exactly as in eval_run.py, and SC* with the labels shuffled. Sparse units can look
                            class-specific by chance (a unit firing on 5 images, 3 of them drusen, scores 60%).
                            The shuffled value is that chance level; sc_star_excess = sc_star - sc_star_shuf.
    ms, ms_floor, ms_star   MS as in eval_run.py (first 2000 validation images, DINOv2 ViT-B/14), the MS of a unit
                            with no preference (= mean pairwise DINOv2 similarity of the images, the exact expectation
                            under shuffled activations), and ms_star = (ms - ms_floor) / (1 - ms_floor).
    purity_proj / _bb       top-k class purity: for each unit, the share of the majority class among the k images that
                            activate it most (highest values). Uses activation strength, so it works for signed SimCLR
                            outputs as well as NCL's. _proj = projector output z (after ReLU for NCL), _bb = backbone
                            feature h (non-negative for both objectives). Units whose activation exceeds its minimum on
                            fewer than k images are left out (count in eligible_*).
    purity_*_shuf, _star    the same with shuffled labels (chance purity), and (purity - shuf) / (1 - shuf).
    random-init rows        untrained ResNet-50 + projector (ReLU for the 'ncl' control): what the architecture gives
                            before any training.
"""
import argparse, csv, hashlib, re, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, ".")

SPEC_VERSION = "s2"   # s2: training-matched readout of z; MS on a fixed class-balanced subset
FIELDS = [
    "run_id", "spec_version", "eval_date", "part", "objective", "policy_id", "seed", "ckpt_path",
    "n_per_class", "draws", "k",
    "sc_star", "sc_star_shuf", "sc_star_excess",
    "ms", "ms_floor", "ms_star",
    "purity_proj", "purity_proj_shuf", "purity_proj_star", "eligible_proj",
    "purity_bb", "purity_bb_shuf", "purity_bb_star", "eligible_bb",
    "script_md5",
]


# ---------------------------------------------------------------- metrics (CPU, plain tensors)
def balanced_index(y, seed, n_classes):
    g = torch.Generator().manual_seed(seed)
    n = min(int((y == c).sum()) for c in range(n_classes))
    idx = torch.cat([(y == c).nonzero(as_tuple=True)[0][torch.randperm(int((y == c).sum()), generator=g)[:n]]
                     for c in range(n_classes)])
    return idx, n


def sc_percent(Zb, yb, relu, n_classes, eps=1e-5):
    """Same computation as eval_run.sc_scores, on an already balanced subset. Returns mean SC in %."""
    alive = (Zb.sum(0) > 0) if relu else (Zb.abs().sum(0) > 0)
    if int(alive.sum()) == 0:
        return float("nan")
    Zn = F.normalize(Zb[:, alive], dim=1)
    fire = (Zn.abs() > eps).float()
    onehot = F.one_hot(yb, n_classes).float()
    counts = onehot.T @ fire
    tot = counts.sum(0)
    ok = tot > 0
    return 100 * (counts.max(0).values[ok] / tot[ok]).mean().item()


def sc_star_of(sc, n_classes):
    return 100 * ((sc / 100 - 1 / n_classes) / (1 - 1 / n_classes))


def topk_purity(A, yb, k, n_classes):
    """Mean over eligible units of the majority-class share among each unit's k highest activations.
    A unit is eligible when more than k images exceed its minimum value (so its top k are not ties at the floor)."""
    k = min(k, A.shape[0])
    above = (A > A.min(0, keepdim=True).values).sum(0)
    elig = above >= k
    if int(elig.sum()) == 0:
        return float("nan"), 0
    top = A[:, elig].topk(k, dim=0).indices                      # k x U
    lab = yb[top]                                                # k x U
    counts = F.one_hot(lab, n_classes).sum(0)                    # U x C
    return (counts.max(1).values.float() / k).mean().item(), int(elig.sum())


def _ms_vectorised(A, S):
    """Copy of eval_run.ms_vectorised, used only by --selftest (main() uses eval_run's own function)."""
    lo, hi = A.min(0).values, A.max(0).values
    live = hi > lo
    An = (A[:, live] - lo[live]) / (hi[live] - lo[live])
    s1 = An.sum(0); s2 = (An ** 2).sum(0)
    num = (An * (S @ An)).sum(0) - s2
    den = s1 ** 2 - s2
    ok = den > 0
    return (num[ok] / den[ok]), A.shape[1] - int(ok.sum())


def balanced_fixed(y, per_class, n_classes, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(y == c).nonzero(as_tuple=True)[0][torch.randperm(int((y == c).sum()), generator=g)[:per_class]]
                      for c in range(n_classes)])


def dinov2_embeddings_idx(idx, cache_dir, tag, device="cuda"):
    """DINOv2 ViT-B/14 embeddings (unit norm) of the validation images at idx, cached under <cache_dir>/dinov2_vitb14_<tag>.pt."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    import eval_run as ER
    cache = Path(cache_dir) / f"dinov2_vitb14_{tag}.pt"
    if cache.exists():
        d = torch.load(cache)
        if torch.equal(d["idx"], idx):
            return d["emb"]
    enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device).eval()
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    sub = Subset(ER.DS(split="val", transform=tf), idx.tolist())
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(sub, batch_size=128, shuffle=False, num_workers=0):
            out.append(enc(x.to(device)).cpu().float())
    emb = F.normalize(torch.cat(out), dim=1)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"idx": idx, "emb": emb}, cache)
    del enc
    return emb


def readout(z_raw, ncl):
    """Projector output as the loss sees it: SimCLR l2-normalised; NCL normalise -> ReLU -> normalise."""
    z = F.normalize(z_raw, dim=1)
    return F.normalize(F.relu(z), dim=1) if ncl else z


def ms_floor_of(S):
    n = S.shape[0]
    return float((S.sum() - S.diagonal().sum()) / (n * (n - 1)))


def audit(h, z, y, relu, n_classes, k=100, draws=5, emb_S=None, ms_fn=None, ms_idx=None):
    """All audit measures for one run from validation features h (N x 2048), z (N x 256) and labels y."""
    sc, sc_sh, p_pj, p_pj_sh, p_bb, p_bb_sh, e_pj, e_bb = [], [], [], [], [], [], [], []
    n = None
    for s in range(draws):
        idx, n = balanced_index(y, s, n_classes)
        Zb, Hb, yb = z[idx], h[idx], y[idx]
        g = torch.Generator().manual_seed(1000 + s)
        ysh = yb[torch.randperm(len(yb), generator=g)]
        sc.append(sc_percent(Zb, yb, relu, n_classes)); sc_sh.append(sc_percent(Zb, ysh, relu, n_classes))
        a, e = topk_purity(Zb, yb, k, n_classes); p_pj.append(a); e_pj.append(e)
        p_pj_sh.append(topk_purity(Zb, ysh, k, n_classes)[0])
        a, e = topk_purity(Hb, yb, k, n_classes); p_bb.append(a); e_bb.append(e)
        p_bb_sh.append(topk_purity(Hb, ysh, k, n_classes)[0])
    def m(v):
        v = [x for x in v if x == x]                              # drop NaN
        return float(np.mean(v)) if v else float("nan")
    out = dict(n_per_class=n, draws=draws, k=k)
    s_star, s_star_sh = sc_star_of(m(sc), n_classes), sc_star_of(m(sc_sh), n_classes)
    out.update(sc_star=s_star, sc_star_shuf=s_star_sh, sc_star_excess=s_star - s_star_sh)
    for name, p, psh, e in (("proj", p_pj, p_pj_sh, e_pj), ("bb", p_bb, p_bb_sh, e_bb)):
        pm, psm = m(p), m(psh)
        out[f"purity_{name}"] = pm
        out[f"purity_{name}_shuf"] = psm
        out[f"purity_{name}_star"] = (pm - psm) / (1 - psm) if psm < 1 else float("nan")
        out[f"eligible_{name}"] = int(round(m(e)))
    if emb_S is not None and ms_fn is not None:
        A = (z[ms_idx] if ms_idx is not None else z[: emb_S.shape[0]]).to(emb_S.device)
        v, _ = ms_fn(A, emb_S)
        floor = ms_floor_of(emb_S)
        ms = v.mean().item() if len(v) else float("nan")
        out.update(ms=ms, ms_floor=floor, ms_star=(ms - floor) / (1 - floor))
    return out


# ---------------------------------------------------------------- bookkeeping
def script_md5():
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]


def latest_master_rows(path):
    latest = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            latest[r["run_id"]] = r
    return list(latest.values())


def done_ids(path):
    if not Path(path).exists():
        return set()
    with open(path) as f:
        return {r["run_id"] for r in csv.DictReader(f) if r.get("spec_version") == SPEC_VERSION}


def append(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k in FIELDS})


def random_model(seed):
    from torchvision import models
    torch.manual_seed(seed)
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256))
    return backbone.cuda().eval(), projector.cuda().eval()


# ---------------------------------------------------------------- self-test on synthetic data
def selftest():
    torch.manual_seed(0)
    C, n = 4, 800
    y = torch.arange(C).repeat_interleave(n)
    N = len(y)
    h = torch.rand(N, 64)                                         # dense non-negative, no class signal
    cases = {}
    # 1) selective sparse units: unit d fires on its class only
    z = torch.zeros(N, 32)
    for d in range(32):
        z[y == d % C, d] = torch.rand(int((y == d % C).sum())) + 0.1
    cases["selective, non-negative"] = (z, True)
    # 2) very sparse random units: each fires on 5 random images (looks selective by chance)
    z = torch.zeros(N, 32)
    for d in range(32):
        z[torch.randperm(N)[:5], d] = 1.0
    cases["sparse random (5 images/unit)"] = (z, True)
    # 3) dense signed random (SimCLR-like with no class signal)
    cases["dense signed random"] = (torch.randn(N, 32), False)
    # 4) dense signed with class signal in magnitude
    z = torch.randn(N, 32)
    for d in range(32):
        z[y == d % C, d] += 2.0
    cases["dense signed, class-shifted"] = (z, False)
    print(f"{'case':32s} {'SC*':>7s} {'SC*shuf':>8s} {'excess':>7s} {'pur':>6s} {'purshuf':>8s} {'pur*':>6s} {'elig':>5s}")
    res = {}
    for name, (z, relu) in cases.items():
        o = audit(h, z, y, relu, C, k=100, draws=3)
        res[name] = o
        print(f"{name:32s} {o['sc_star']:7.1f} {o['sc_star_shuf']:8.1f} {o['sc_star_excess']:7.1f} "
              f"{o['purity_proj']:6.3f} {o['purity_proj_shuf']:8.3f} {o['purity_proj_star']:6.3f} {o['eligible_proj']:5d}")
    assert res["selective, non-negative"]["sc_star_excess"] > 80
    assert res["selective, non-negative"]["purity_proj_star"] > 0.95
    assert abs(res["sparse random (5 images/unit)"]["sc_star_excess"]) < 15      # high SC*, but chance explains it
    assert res["sparse random (5 images/unit)"]["sc_star"] > 20
    assert abs(res["dense signed random"]["sc_star"]) < 1                         # floor, as SimCLR
    assert abs(res["dense signed random"]["purity_proj_star"]) < 0.05
    assert res["dense signed, class-shifted"]["purity_proj_star"] > 0.5            # purity sees magnitude signal
    assert abs(res["dense signed, class-shifted"]["sc_star"]) < 1                  # SC* cannot
    # MS floor: random activations score at the floor
    E = F.normalize(torch.randn(300, 16) + 2.0, dim=1)
    S = E @ E.T
    A = torch.rand(300, 50)
    v, _ = _ms_vectorised(A, S)
    floor = ms_floor_of(S)
    print(f"MS of random units {v.mean():.4f}  vs floor {floor:.4f}")
    assert abs(v.mean().item() - floor) < 0.01
    print("selftest passed")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="octmnist", help="octmnist or pathmnist")
    ap.add_argument("--master", default=None, help="default: <study folder of the dataset>/results_master.csv")
    ap.add_argument("--out", default=None, help="default: <study folder of the dataset>/results_spec.csv")
    ap.add_argument("--only", default=".", help="regex on run_id")
    ap.add_argument("--random-init", type=int, default=0, help="number of untrained control networks per objective")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--cache", default="study/cache")
    ap.add_argument("--ms-per-class", type=int, default=500, help="MS subset: images per class (fixed, seed 0)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return

    import eval_run as ER
    ER.set_dataset(a.dataset)
    a.master = a.master or f"{ER.STUDY_DIR}/results_master.csv"
    a.out = a.out or f"{ER.STUDY_DIR}/results_spec.csv"
    print(f"dataset {ER.DATASET} ({ER.N_CLASSES} classes); reading {a.master}, writing {a.out}")
    done = set() if a.force else done_ids(a.out)
    todo = [r for r in latest_master_rows(a.master) if re.search(a.only, r["run_id"]) and r["run_id"] not in done]
    controls = []
    for obj in ("ncl", "simclr"):
        for s in range(a.random_init):
            rid = f"random-init-{obj}-s{s}"
            if rid not in done:
                controls.append(dict(run_id=rid, objective=obj, policy_id="random-init", seed=s, part="control", ckpt_path=""))
    print(f"{len(todo)} trained run(s) and {len(controls)} control(s) to audit")
    if not todo and not controls:
        return

    S, ms_idx = None, None
    try:
        yv = torch.as_tensor(ER.DS(split="val").labels)
        per = min(a.ms_per_class, min(int((yv == c).sum()) for c in range(ER.N_CLASSES)))
        ms_idx = balanced_fixed(yv, per, ER.N_CLASSES, seed=0)
        prefix = "" if ER.DATASET == "octmnist" else f"{ER.DATASET}_"
        emb = dinov2_embeddings_idx(ms_idx, a.cache, f"{prefix}valbal{per}x{ER.N_CLASSES}")
        S = (emb @ emb.T).cuda()
        print(f"MS subset: {len(ms_idx)} validation images ({per} per class); "
              f"MS floor (mean pairwise DINOv2 similarity): {ms_floor_of(S):.4f}")
    except Exception as e:
        print(f"WARNING: DINOv2 embeddings unavailable ({e!r}); MS columns left empty", flush=True)

    for r in todo + controls:
        t0 = time.time()
        relu = r["objective"] == "ncl"
        if r["policy_id"] == "random-init":
            bb, pj = random_model(int(r["seed"]))
        else:
            bb, pj, _ = ER.load_model(r["ckpt_path"])
        h, z_raw, y = ER.extract(bb, pj, "val", False, a.bs, a.workers)   # raw projector output
        z = readout(z_raw, relu)                                            # as the loss sees it
        o = audit(h, z, y, relu, ER.N_CLASSES, k=a.k, draws=a.draws, emb_S=S, ms_fn=ER.ms_vectorised, ms_idx=ms_idx)
        row = dict(run_id=r["run_id"], spec_version=SPEC_VERSION, eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
                   part=r.get("part", ""), objective=r["objective"], policy_id=r["policy_id"], seed=r.get("seed", ""),
                   ckpt_path=r.get("ckpt_path", ""), script_md5=script_md5(), **o)
        append(a.out, row)
        ms_txt = f"MS* {o['ms_star']:.3f}" if "ms_star" in o else "MS* -"
        print(f"{r['run_id']:44s} SC* {o['sc_star']:6.2f} (shuf {o['sc_star_shuf']:5.2f})  {ms_txt}  "
              f"purity proj {o['purity_proj_star']:.3f}  bb {o['purity_bb_star']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        del bb, pj, h, z, z_raw
        torch.cuda.empty_cache()
    print(f"\nappended to {a.out}")


if __name__ == "__main__":
    main()
