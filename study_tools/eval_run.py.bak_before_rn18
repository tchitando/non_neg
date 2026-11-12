#!/usr/bin/env python3
"""Score finished study runs and append one row per run to the master results table.

Run from the repository root on the server (the folder that contains MedMNIST/ and solo/):

    python study_tools/eval_run.py --list                       # run id -> checkpoint, nothing evaluated
    python study_tools/eval_run.py --manifest study/manifest_part1.csv --only '^ncl-.*-(blur|motion)$'
    python study_tools/eval_run.py --manifest study/manifest_part1.csv        # every finished, unscored run

Only the TRAIN and VALIDATION splits are read. The test split is never touched here (it is opened once,
by a separate script, after the confirmation seeds).

What is measured, per run (all on the validation split unless noted):
    probe_bb_bal_acc    linear probe on the 2048-d backbone features, balanced accuracy  <- the deciding score
    probe_proj_bal_acc  linear probe on the 256-d projector features (post-ReLU for NCL), as in E5/E6
    knn{5,20,100}_bal_acc  k-NN on L2-normalised backbone features, train as the bank
    sc, sc_star         class selectivity of the projector units on a class-balanced validation subset
                        (mean over alive units of the share of the top class among images that activate the unit;
                         SC* = (SC - 1/C) / (1 - 1/C)), averaged over 5 balancing seeds
    ms, ms_std, ms_max  monosemanticity score, DINOv2 ViT-B/14 embeddings of the first 2000 validation images,
                        same definition as Mono_octmnist.py (vectorised; checked against the loop version)
    alive_proj, dead_proj   projector units that are ever non-zero on the validation set
    effdim_proj, effdim_bb  participation ratio of the centred features (as eval_mechanistic_octmnist.py)

The master table study/results_master.csv is append-only: existing rows are never edited or removed.
A run that is already scored (same run_id and eval_version) is skipped unless --force is given, in which case
a new row is appended and the later row is the one the derived tables use.
"""
import argparse, csv, glob, hashlib, json, math, os, re, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, ".")
import importlib

EVAL_VERSION = "v1"
# Dataset registry: module, dataset class, mean, std, number of classes, study folder (manifests and result tables).
DATASETS = {
    "octmnist": ("MedMNIST.octmnist_dataset", "OCTMnistDataset", "OCTMNIST_MEAN", "OCTMNIST_STD", 4, "study"),
    "pathmnist": ("MedMNIST.pathmnist_dataset", "PathMnistDataset", "PATHMNIST_MEAN", "PATHMNIST_STD", 9, "study/path"),
}
DATASET, DS, MEAN, STD, N_CLASSES, STUDY_DIR = None, None, None, None, None, None


def set_dataset(name):
    """Select the dataset; every function below reads these module-level names when it is called."""
    global DATASET, DS, MEAN, STD, N_CLASSES, STUDY_DIR
    mod, cls, mean, std, n, study = DATASETS[name]
    m = importlib.import_module(mod)
    DATASET, DS, MEAN, STD, N_CLASSES, STUDY_DIR = name, getattr(m, cls), getattr(m, mean), getattr(m, std), n, study


set_dataset("octmnist")      # default: the OCT study, unchanged
MASTER_FIELDS = [
    "run_id", "eval_version", "eval_date", "part", "objective", "seed", "policy_id", "bitmask", "augs_on",
    "n_augs", "prob", "levels", "crop_min_scale", "ckpt_path", "ckpt_epoch", "n_ckpt_matches",
    "probe_bb_bal_acc", "probe_bb_acc", "probe_proj_bal_acc", "probe_proj_acc",
    "knn5_bal_acc", "knn20_bal_acc", "knn100_bal_acc",
    "sc", "sc_star", "ms", "ms_std", "ms_max", "alive_proj", "dead_proj", "effdim_proj", "effdim_bb",
    "probe_opt", "split", "eval_script_md5", "git_head",
]


# ---------------------------------------------------------------- bookkeeping
def script_md5():
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def read_manifests(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def find_ckpts(run_id, root):
    """Checkpoints are written to <root>/**/<run_id>-offline-<wandb id>-ep=99.ckpt (wandb makes the folder name)."""
    pat = os.path.join(root, "**", f"{run_id}-offline-*-ep=99.ckpt")
    hits = glob.glob(pat, recursive=True)
    return sorted(hits, key=os.path.getmtime)


def master_done(path):
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for r in csv.DictReader(f):
                done.add((r["run_id"], r["eval_version"]))
    return done


def append_row(path, row):
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MASTER_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in MASTER_FIELDS})


# ---------------------------------------------------------------- model + features
def load_model(ckpt_path):
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    projector = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 256))
    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck["state_dict"]
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}, strict=True)
    projector.load_state_dict({k[len("projector."):]: v for k, v in sd.items() if k.startswith("projector.")}, strict=True)
    return backbone.cuda().eval(), projector.cuda().eval(), ck.get("epoch", "")


@torch.no_grad()
def extract(backbone, projector, split, relu, bs, workers):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    dl = DataLoader(DS(split=split, transform=tf), batch_size=bs, shuffle=False, num_workers=workers)
    hs, zs, ys = [], [], []
    for x, y in dl:
        h = backbone(x.cuda())
        z = projector(h)
        if relu:
            z = F.relu(z)
        hs.append(h.cpu()); zs.append(z.cpu()); ys.append(torch.as_tensor(y))
    return torch.cat(hs), torch.cat(zs), torch.cat(ys)


# ---------------------------------------------------------------- probe, kNN
def linear_probe(Xtr, ytr, Xva, yva, seed=0, epochs=50, lr=0.1, wd=1e-4, warmup=10, bs=256):
    """Same recipe as eval_linear_octmnist.py (LARS, warm-up + cosine, 50 epochs). Falls back to SGD if solo-learn is absent."""
    torch.manual_seed(seed); np.random.seed(seed)
    clf = nn.Linear(Xtr.shape[1], N_CLASSES).cuda()
    try:
        from solo.utils.lars import LARS
        opt = LARS(clf.parameters(), lr=lr, weight_decay=wd, clip_lr=True, eta=0.02, exclude_bias_n_norm=True)
        opt_name = "LARS"
    except Exception:
        opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        opt_name = "SGD(fallback)"
    spe = math.ceil(len(Xtr) / bs)
    total, warm = epochs * spe, warmup * spe

    def lam(step):
        if step < warm:
            return step / max(warm, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lam)
    Xg, yg = Xtr.cuda(), ytr.cuda()
    for _ in range(epochs):
        clf.train()
        perm = torch.randperm(len(Xg), device=Xg.device)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = F.cross_entropy(clf(Xg[idx]), yg[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    clf.eval()
    with torch.no_grad():
        pred = clf(Xva.cuda()).argmax(1).cpu().numpy()
    y = yva.numpy()
    return 100 * balanced_accuracy_score(y, pred), 100 * float((pred == y).mean()), opt_name


@torch.no_grad()
def knn_bal_acc(Xtr, ytr, Xva, yva, ks=(5, 20, 100)):
    bank = F.normalize(Xtr, dim=1).cuda(); q = F.normalize(Xva, dim=1).cuda(); yb = ytr.cuda()
    out = {}
    preds = {k: [] for k in ks}
    for i in range(0, len(q), 2048):
        sim = q[i:i + 2048] @ bank.T
        top_s, top_i = sim.topk(max(ks), dim=1)
        for k in ks:
            sc = torch.zeros(sim.size(0), N_CLASSES, device=sim.device)
            sc.scatter_add_(1, yb[top_i[:, :k]], top_s[:, :k])
            preds[k].append(sc.argmax(1).cpu())
    for k in ks:
        out[k] = 100 * balanced_accuracy_score(yva.numpy(), torch.cat(preds[k]).numpy())
    return out


# ---------------------------------------------------------------- SC, SC*, alive, effective dimension
def sc_scores(Z, y, relu, seed, eps=1e-5):
    """Per-unit class selectivity on a class-balanced subset. Returns (mean SC over alive units in %, n_alive)."""
    g = torch.Generator().manual_seed(seed)
    n = min(int((y == c).sum()) for c in range(N_CLASSES))
    idx = torch.cat([(y == c).nonzero(as_tuple=True)[0][torch.randperm(int((y == c).sum()), generator=g)[:n]]
                     for c in range(N_CLASSES)])
    Zb, yb = Z[idx], y[idx]
    alive = (Zb.sum(0) > 0) if relu else (Zb.abs().sum(0) > 0)
    Zn = F.normalize(Zb[:, alive], dim=1)
    fire = (Zn.abs() > eps).float()                                   # N x alive
    onehot = F.one_hot(yb, N_CLASSES).float()                         # N x C
    counts = onehot.T @ fire                                          # C x alive
    tot = counts.sum(0)
    ok = tot > 0
    share = (counts.max(0).values[ok] / tot[ok])
    return 100 * share.mean().item(), int(alive.sum())


def eff_dim(F_):
    f = F_.double() - F_.double().mean(0)
    cov = (f.T @ f) / len(f)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    lam = lam[lam > 1e-10]
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if len(lam) else 0.0


# ---------------------------------------------------------------- MS (DINOv2)
def dinov2_embeddings(n_images, cache_dir, device="cuda"):
    tag = "" if DATASET == "octmnist" else f"{DATASET}_"
    cache = Path(cache_dir) / f"dinov2_vitb14_{tag}val{n_images}.pt"
    if cache.exists():
        return torch.load(cache)
    enc = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device).eval()
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = DS(split="val", transform=tf)
    sub = torch.utils.data.Subset(ds, list(range(min(n_images, len(ds)))))
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(sub, batch_size=128, shuffle=False, num_workers=0):
            out.append(enc(x.to(device)).cpu().float())
    emb = F.normalize(torch.cat(out), dim=1)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache)
    del enc
    return emb


def ms_vectorised(A, S):
    """MS_k = sum_{i!=j} a_i a_j S_ij / sum_{i!=j} a_i a_j with a = min-max scaled activation of unit k (Pach et al.).
    Needs S_ii = 1 (unit-norm embeddings). Returns per-unit MS for the non-dead units and the dead count."""
    lo, hi = A.min(0).values, A.max(0).values
    live = hi > lo
    An = (A[:, live] - lo[live]) / (hi[live] - lo[live])
    s1 = An.sum(0); s2 = (An ** 2).sum(0)
    num = (An * (S @ An)).sum(0) - s2
    den = s1 ** 2 - s2
    ok = den > 0
    return (num[ok] / den[ok]), A.shape[1] - int(ok.sum())


def ms_loop(A, S):
    """Reference implementation copied from Mono_octmnist.py, used only by --selftest."""
    N, D = A.shape
    off = ~torch.eye(N, dtype=torch.bool)
    out, dead = [], 0
    for k in range(D):
        a = A[:, k]
        if a.max() == a.min():
            dead += 1; continue
        an = (a - a.min()) / (a.max() - a.min())
        R = an.unsqueeze(1) * an.unsqueeze(0)
        w = R[off].sum()
        if w == 0:
            dead += 1; continue
        out.append(((R[off] * S[off]).sum() / w).item())
    return torch.tensor(out), dead


def selftest():
    torch.manual_seed(0)
    N, D = 200, 12
    E = F.normalize(torch.randn(N, 32), dim=1); S = E @ E.T
    A = torch.relu(torch.randn(N, D)); A[:, 3] = 0; A[:, 5] = 1.0
    v, dv = ms_vectorised(A, S); l, dl = ms_loop(A, S)
    assert dv == dl and torch.allclose(v, l, atol=1e-5), (v, l)
    # SC on a toy case: unit k fires only for class k%4 -> SC = 100
    y = torch.arange(400) % 4
    Z = torch.zeros(400, 8)
    for k in range(8):
        Z[y == k % 4, k] = 1.0
    sc, alive = sc_scores(Z, y, relu=True, seed=0)
    assert abs(sc - 100) < 1e-6 and alive == 8, (sc, alive)
    # effective dimension: identity covariance on 5 dims -> 5
    X = torch.randn(20000, 5)
    assert abs(eff_dim(X) - 5) < 0.2, eff_dim(X)
    print("selftest OK: vectorised MS equals the loop version, SC and effective dimension behave on toy cases")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="octmnist", choices=sorted(DATASETS))
    ap.add_argument("--manifest", nargs="*", default=None,
                    help="manifest csv files (default: OCT study/manifest_part*.csv; PathMNIST study/path/manifest.csv)")
    ap.add_argument("--only", default=".", help="regex on run_id")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--master", default=None, help="default: <study folder of the dataset>/results_master.csv")
    ap.add_argument("--cache", default="study/cache")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ms-images", type=int, default=2000)
    ap.add_argument("--sc-seeds", type=int, default=5)
    ap.add_argument("--list", action="store_true", help="only print run id -> checkpoint")
    ap.add_argument("--require-done", action="store_true", help="only runs listed in study/done_part*.txt")
    ap.add_argument("--force", action="store_true", help="score again and append a new row")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    set_dataset(a.dataset)
    a.master = a.master or f"{STUDY_DIR}/results_master.csv"
    Path(a.master).parent.mkdir(parents=True, exist_ok=True)
    print(f"dataset {DATASET} ({N_CLASSES} classes); results table {a.master}")

    default_mans = sorted(glob.glob("study/manifest_part*.csv")) if DATASET == "octmnist" else [f"{STUDY_DIR}/manifest.csv"]
    mans = a.manifest or default_mans
    rows = [r for r in read_manifests(mans) if re.search(a.only, r["run_id"])]
    done_runs = set()
    if a.require_done:
        for p in glob.glob("study/done_part*.txt") + glob.glob(f"{STUDY_DIR}/done.txt"):
            done_runs |= set(Path(p).read_text().split())
    scored = master_done(a.master)
    todo = []
    for r in rows:
        hits = find_ckpts(r["run_id"], a.ckpt_root)
        r["_hits"] = hits
        status = "no checkpoint yet" if not hits else ("scored" if (r["run_id"], EVAL_VERSION) in scored and not a.force else "ready")
        if a.require_done and r["run_id"] not in done_runs:
            status = "not in done list"
        warn = f"  ({len(hits)} checkpoints, using the newest)" if len(hits) > 1 else ""
        print(f"{r['run_id']:44s} {status}{warn}")
        if status == "ready":
            todo.append(r)
    if a.list:
        for r in rows:
            for h in r["_hits"]:
                print("  ", r["run_id"], "->", h)
        return
    print(f"\n{len(todo)} run(s) to score\n")
    if not todo:
        return

    torch.backends.cudnn.benchmark = False
    emb = None
    try:
        emb = dinov2_embeddings(a.ms_images, a.cache)
        S = (emb @ emb.T).cuda()
    except Exception as e:
        print(f"WARNING: DINOv2 embeddings unavailable ({e!r}); MS will be left empty", flush=True)

    for r in todo:
        t0 = time.time()
        ckpt = r["_hits"][-1]
        relu = r["objective"] == "ncl"
        print(f"=== {r['run_id']}  {ckpt}", flush=True)
        bb, pj, ep = load_model(ckpt)
        htr, ztr, ytr = extract(bb, pj, "train", relu, a.bs, a.workers)
        hva, zva, yva = extract(bb, pj, "val", relu, a.bs, a.workers)
        p_bb = linear_probe(htr, ytr, hva, yva)
        p_pj = linear_probe(ztr, ytr, zva, yva)
        knn = knn_bal_acc(htr, ytr, hva, yva)
        scs = [sc_scores(zva, yva, relu, seed=s) for s in range(a.sc_seeds)]
        sc = float(np.mean([x[0] for x in scs]))
        sc_star = 100 * ((sc / 100 - 1 / N_CLASSES) / (1 - 1 / N_CLASSES))
        alive = (zva.sum(0) > 0) if relu else (zva.abs().sum(0) > 0)
        ms = ms_std = ms_max = ""
        if emb is not None:
            A = zva[: emb.shape[0]].cuda()
            v, _ = ms_vectorised(A, S)
            if len(v):
                ms, ms_std, ms_max = round(v.mean().item(), 5), round(v.std().item(), 5), round(v.max().item(), 5)
        row = dict(r)
        row.update(
            eval_version=EVAL_VERSION, eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            ckpt_path=ckpt, ckpt_epoch=ep, n_ckpt_matches=len(r["_hits"]),
            probe_bb_bal_acc=round(p_bb[0], 3), probe_bb_acc=round(p_bb[1], 3),
            probe_proj_bal_acc=round(p_pj[0], 3), probe_proj_acc=round(p_pj[1], 3),
            knn5_bal_acc=round(knn[5], 3), knn20_bal_acc=round(knn[20], 3), knn100_bal_acc=round(knn[100], 3),
            sc=round(sc, 3), sc_star=round(sc_star, 3), ms=ms, ms_std=ms_std, ms_max=ms_max,
            alive_proj=int(alive.sum()), dead_proj=int((~alive).sum()),
            effdim_proj=round(eff_dim(zva), 2), effdim_bb=round(eff_dim(hva), 2),
            probe_opt=p_bb[2], split="val", eval_script_md5=script_md5(), git_head=git_head(),
        )
        append_row(a.master, row)
        print(f"    probe(bb) {row['probe_bb_bal_acc']}  probe(proj) {row['probe_proj_bal_acc']}  knn20 {row['knn20_bal_acc']}  "
              f"SC {row['sc']}  SC* {row['sc_star']}  MS {row['ms']}  alive {row['alive_proj']}/{zva.shape[1]}  "
              f"effdim {row['effdim_proj']}  ({time.time() - t0:.0f}s)", flush=True)
        del bb, pj, htr, ztr, hva, zva
        torch.cuda.empty_cache()
    print(f"\nappended to {a.master}")


if __name__ == "__main__":
    main()
