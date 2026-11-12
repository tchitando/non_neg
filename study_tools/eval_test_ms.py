#!/usr/bin/env python3
"""MS on the test split, for runs in a test plan. Written 3 Nov 2026.

eval_spec.py computes MS on validation only (a fixed class-balanced subset, 500 images per class, seed 0). This
script computes the same MS on a fixed class-balanced TEST subset, so specialisation can be reported on the same
images as the test probes:

    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_test_ms.py --plan study/test_plan_oct.csv --list     # status only
    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_test_ms.py --plan study/test_plan_oct.csv study/test_plan_croponly.csv
    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_test_ms.py --plan study/test_plan_path.csv
    CUDA_VISIBLE_DEVICES=1 python study_tools/eval_test_ms.py --plan study/test_plan_oct.csv --check 2

Rules, as in eval_test.py: only runs listed in a test plan are scored; a run already in the output file is never
scored again; runs without an epoch-99 checkpoint are skipped (run the command again later). The checkpoint is the
one eval_test.py used for the run's test probes (read from study/test_results_<dataset>.csv) when that file has
the run, so MS and the test probes come from the same weights.

The computation is eval_spec.py's, with the split changed:
    z               projector output read out as the loss sees it (eval_spec.readout: SimCLR l2-normalised;
                    NCL normalise -> ReLU -> normalise)
    subset          eval_spec.balanced_fixed with seed 0, n per class = min(--ms-per-class, smallest test class):
                    OCTMNIST 250 per class (all 1,000 test images), PathMNIST 339 per class (smallest class)
    embeddings      DINOv2 ViT-B/14, unit norm, same transform as eval_spec.py; cached in --cache
    ms              eval_run.ms_vectorised over units that are not dead on the subset, then the mean over units
    ms_floor        mean pairwise DINOv2 similarity of the subset (eval_spec.ms_floor_of)
    ms_star         (ms - ms_floor) / (1 - ms_floor)
The floor differs from validation's (other images, other subset size), so compare test MS with test MS, and use
ms_star when comparing the two splits.

--check N recomputes VALIDATION MS for the first N planned runs with this script's code path and prints it next to
the value in results_spec.csv (nothing is written). Matching values show that the code path reproduces eval_spec.py.

Results are appended to --out (default: study/test_ms_<dataset>.csv).
"""
import argparse, csv, hashlib, sys, time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent

FIELDS = [
    "run_id", "dataset", "backbone", "objective", "seed", "policy_id", "ckpt_path", "ckpt_source", "split",
    "n_per_class", "n_images", "ms", "ms_std", "ms_max", "ms_floor", "ms_star", "alive_units", "dead_units",
    "plan", "eval_date", "script_md5",
]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_dinov2(device="cuda"):
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True).to(device).eval()


def dinov2_embeddings_split(ER, idx, split, cache_dir, tag, device="cuda", encoder_fn=load_dinov2):
    """eval_spec.dinov2_embeddings_idx with the split as an argument (that function reads validation only)."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset
    cache = Path(cache_dir) / f"dinov2_vitb14_{tag}.pt"
    if cache.exists():
        d = torch.load(cache)
        if torch.equal(d["idx"], idx):
            return d["emb"]
    enc = encoder_fn(device)
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    sub = Subset(ER.DS(split=split, transform=tf), idx.tolist())
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(sub, batch_size=128, shuffle=False, num_workers=0):
            out.append(enc(x.to(device)).cpu().float())
    emb = F.normalize(torch.cat(out), dim=1)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"idx": idx, "emb": emb}, cache)
    del enc
    return emb


def subset_and_S(ER, ES, split, per_max, cache_dir, device="cuda", encoder_fn=load_dinov2):
    """Fixed class-balanced subset of a split (seed 0) and the DINOv2 similarity matrix of its images."""
    y = torch.as_tensor(ER.DS(split=split).labels)
    C = ER.N_CLASSES
    per = min(per_max, min(int((y == c).sum()) for c in range(C)))
    idx = ES.balanced_fixed(y, per, C, seed=0)
    prefix = "" if ER.DATASET == "octmnist" else f"{ER.DATASET}_"
    tag = f"{prefix}{'valbal' if split == 'val' else split + 'bal'}{per}x{C}"   # validation tag = eval_spec.py's cache
    emb = dinov2_embeddings_split(ER, idx, split, cache_dir, tag, device, encoder_fn)
    return idx, per, (emb @ emb.T).to(device)


def ms_of_run(ER, ES, ckpt, ncl, split, idx, S, bs, workers):
    bb, pj, _ = ER.load_model(ckpt)
    _, z_raw, _ = ER.extract(bb, pj, split, False, bs, workers)        # raw projector output, every image of the split
    z = ES.readout(z_raw, ncl)                                          # as the loss sees it (as eval_spec.py)
    v, dead = ER.ms_vectorised(z[idx].to(S.device), S)
    out = dict(ms=v.mean().item() if len(v) else float("nan"),
               ms_std=v.std().item() if len(v) > 1 else float("nan"),
               ms_max=v.max().item() if len(v) else float("nan"),
               alive_units=int(len(v)), dead_units=int(dead))
    del bb, pj, z, z_raw
    return out


def main(argv=None, encoder_fn=load_dinov2):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", required=True, help="test plan csv file(s), all for the same dataset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--cache", default="study/cache")
    ap.add_argument("--ms-per-class", type=int, default=500, help="upper limit; the smallest test class sets it")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="only show which planned runs are trained / scored")
    ap.add_argument("--check", type=int, default=0, help="recompute validation MS for N runs and compare (no writing)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(HERE)); sys.path.insert(0, ".")
    import eval_run as ER
    import eval_spec as ES

    plan, seen = [], set()
    for p in a.plan:
        for r in read_csv(p):
            if r["run_id"] not in seen:
                seen.add(r["run_id"]); r["_plan"] = p; plan.append(r)
    datasets = {r["dataset"] for r in plan}
    if len(datasets) != 1:
        sys.exit(f"one dataset per call, found {sorted(datasets)}")
    ds = datasets.pop()
    ER.set_dataset(ds)
    C = ER.N_CLASSES
    out = Path(a.out or f"study/test_ms_{ds}.csv")
    done = {r["run_id"] for r in read_csv(out)} if out.exists() else set()
    tres = Path(f"study/test_results_{ds}.csv")
    test_ckpt = {r["run_id"]: r["ckpt_path"] for r in read_csv(tres)} if tres.exists() else {}

    todo = []
    for r in plan:
        ck, src = None, ""
        if r["run_id"] in test_ckpt and Path(test_ckpt[r["run_id"]]).exists():
            ck, src = test_ckpt[r["run_id"]], "test_results"
        else:
            hits = ER.find_ckpts(r["run_id"], a.ckpt_root)
            if hits:
                ck, src = hits[-1], "latest epoch-99"
        status = "scored" if r["run_id"] in done else ("not trained yet" if ck is None else "ready")
        print(f"{r['run_id']:58s} {status}")
        r["_ckpt"], r["_src"] = ck, src
        if status == "ready":
            todo.append(r)
    n_tr = sum(1 for r in plan if r["_ckpt"])
    print(f"\n{len(plan)} planned runs ({ds}), {n_tr} trained, {len(done & seen)} scored, {len(todo)} to score now")

    if a.check:
        spec_path = Path(ER.STUDY_DIR) / "results_spec.csv"
        spec = {}
        if spec_path.exists():
            for r in read_csv(spec_path):
                if r.get("spec_version") == ES.SPEC_VERSION:
                    spec[r["run_id"]] = r                               # latest s2 row per run
        idx_v, per_v, S_v = subset_and_S(ER, ES, "val", 500, a.cache, a.device, encoder_fn)
        print(f"\ncheck on validation: {len(idx_v)} images ({per_v} per class), floor {ES.ms_floor_of(S_v):.5f}")
        for r in [r for r in plan if r["_ckpt"]][: a.check]:
            o = ms_of_run(ER, ES, r["_ckpt"], r["objective"] == "ncl", "val", idx_v, S_v, a.bs, a.workers)
            ref = spec.get(r["run_id"], {}).get("ms", "")
            print(f"  {r['run_id']:58s} MS here {o['ms']:.5f}   results_spec.csv {ref or 'n/a'}")
        return
    if a.list or not todo:
        return

    idx, per, S = subset_and_S(ER, ES, "test", a.ms_per_class, a.cache, a.device, encoder_fn)
    floor = ES.ms_floor_of(S)
    print(f"test subset: {len(idx)} images ({per} per class), MS floor {floor:.5f}")
    md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:12]
    new = not out.exists()
    for r in todo:
        t0 = time.time()
        o = ms_of_run(ER, ES, r["_ckpt"], r["objective"] == "ncl", "test", idx, S, a.bs, a.workers)
        row = dict(run_id=r["run_id"], dataset=ds, backbone=r.get("backbone", ""), objective=r["objective"],
                   seed=r["seed"], policy_id=r["policy_id"], ckpt_path=r["_ckpt"], ckpt_source=r["_src"], split="test",
                   n_per_class=per, n_images=len(idx), ms_floor=floor, ms_star=(o["ms"] - floor) / (1 - floor),
                   plan=r["_plan"], eval_date=datetime.now().strftime("%Y-%m-%d %H:%M"), script_md5=md5, **o)
        with out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader(); new = False
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
        print(f"{r['run_id']:58s} MS {o['ms']:.4f}  MS* {row['ms_star']:.4f}  alive {o['alive_units']}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if a.device == "cuda":
            torch.cuda.empty_cache()
    print(f"\nappended to {out}")


if __name__ == "__main__":
    main()
