#!/usr/bin/env python3
"""What has been run and what has not: every run in every manifest, with its training and scoring status.

Run from the Non_neg folder on the server (reads files only, starts nothing):
    python study_tools/run_status.py                        # summary per manifest + what still needs doing
    python study_tools/run_status.py --only 'pixelate|jpeg' # only runs whose id matches the regex
    python study_tools/run_status.py --all                  # also list every run with its status

Training status (one per run, checked in this order):
    running       a main_pretrain.py process with the run's config is alive (ps), or one of its checkpoints
                  was written in the last 30 minutes
    diverged      listed in a nonfinite_runs*.txt by a scorer (features were NaN/inf)
    trained       its epoch-99 checkpoint exists under trained_models/
    failed        claimed in study/claims/ but no epoch-99 checkpoint and not running
                  (crashed or interrupted; rmdir the claim and a launcher will retry it)
    not started   no claim and no checkpoint
Scoring status (trained runs): validation scored (any study/**/results_master.csv or dyn_results.csv),
in a test plan (study/**/test_plan*.csv), test scored (study/**/test_results_*.csv), test MS (study/**/test_ms_*.csv).
Writes study/run_status.csv with one row per run.
"""
import argparse
import csv
import glob
import os
import re
import subprocess
import time
from pathlib import Path


def ids_in(pattern, col="run_id"):
    """Union of the run_id column over every CSV matching the glob pattern."""
    out = set()
    for f in glob.glob(pattern, recursive=True):
        try:
            with open(f, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get(col):
                        out.add(r[col].strip())
        except Exception as e:                      # a half-written file should not stop the report
            print(f"(could not read {f}: {e})")
    return out


def checkpoints(root):
    """run_id -> (highest epoch, newest file time) over <run_id>-offline-<wandb id>-ep=<N>.ckpt files under root."""
    best = {}
    pat = re.compile(r"^(.*)-offline-[^-]+-ep=(\d+)\.ckpt$")
    for dirpath, _, files in os.walk(root):
        for fn in files:
            m = pat.match(fn)
            if m:
                rid, ep = m.group(1), int(m.group(2))
                t = os.path.getmtime(os.path.join(dirpath, fn))
                e0, t0 = best.get(rid, (-1, 0))
                best[rid] = (max(ep, e0), max(t, t0))
    return best


def running_configs():
    """Config names of main_pretrain.py processes that are alive (this user's processes)."""
    try:
        ps = subprocess.run(["ps", "-U", str(os.getuid()), "-o", "args="], capture_output=True, text=True).stdout
    except Exception:
        return set()
    cfgs = set()
    for line in ps.splitlines():
        if "main_pretrain" in line:
            m = re.search(r"--config-name[ =](\S+)", line)
            if m:
                cfgs.add(re.sub(r"\.yaml$", "", m.group(1)))
    return cfgs


def nonfinite():
    out = set()
    for f in glob.glob("study/**/nonfinite_runs*.txt", recursive=True):
        out |= {ln.split("\t")[0].split()[0] for ln in Path(f).read_text().splitlines() if ln.strip()}
    return out


def manifests():
    fs = sorted(glob.glob("study/manifest_part*.csv"), key=lambda f: int(re.search(r"part(\d+)", f).group(1)))
    return fs + sorted(glob.glob("study/*/manifest.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=".", help="regex on run_id")
    ap.add_argument("--all", action="store_true", help="also list every run with its status")
    ap.add_argument("--ckpt-root", default="trained_models")
    ap.add_argument("--csv", default="study/run_status.csv")
    a = ap.parse_args()
    if not Path("study").is_dir():
        raise SystemExit("run this from the Non_neg folder (cd /data/tchitando/Non_neg)")

    ck = checkpoints(a.ckpt_root)
    alive = running_configs()
    bad = nonfinite()
    val = ids_in("study/**/results_master.csv") | ids_in("study/**/dyn_results.csv")
    plan = ids_in("study/**/test_plan*.csv")
    test = ids_in("study/**/test_results_*.csv")
    tms = ids_in("study/**/test_ms_*.csv")
    rx = re.compile(a.only)

    rows, seen = [], set()
    for mf in manifests():
        with open(mf, newline="") as fh:
            for r in csv.DictReader(fh):
                rid = r["run_id"].strip()
                if rid in seen or not rx.search(rid):
                    continue
                seen.add(rid)
                cfg = re.sub(r"\.yaml$", "", r.get("launch_cfg") or r.get("run_yaml") or "")
                claim = Path("study/claims") / rid
                ep, tck = ck.get(rid, (-1, 0))
                if (cfg and cfg in alive) or (ep < 99 and time.time() - tck < 30 * 60):
                    st = "running"                  # live process, or a checkpoint written in the last 30 minutes
                elif rid in bad:
                    st = "diverged"
                elif ep == 99:
                    st = "trained"
                elif claim.is_dir():
                    st = "failed"
                else:
                    st = "not started"
                since = time.strftime("%d %b %H:%M", time.localtime(claim.stat().st_mtime)) if claim.is_dir() else ""
                rows.append(dict(manifest=mf, run_id=rid, status=st, last_ckpt_epoch=ep if ep >= 0 else "",
                                 claimed=since, val_scored=rid in val, in_test_plan=rid in plan,
                                 test_scored=rid in test, test_ms=rid in tms))

    # ---- summary per manifest
    heads = ["manifest", "runs", "trained", "running", "failed", "not started", "diverged",
             "val scored", "in plan", "test scored", "test MS"]
    print("  ".join(f"{h:>11}" if i else f"{h:<26}" for i, h in enumerate(heads)))
    for mf in dict.fromkeys(r["manifest"] for r in rows):
        g = [r for r in rows if r["manifest"] == mf]
        c = lambda s: sum(r["status"] == s for r in g)
        vals = [len(g), c("trained"), c("running"), c("failed"), c("not started"), c("diverged"),
                sum(r["val_scored"] for r in g), sum(r["in_test_plan"] for r in g),
                sum(r["test_scored"] for r in g), sum(r["test_ms"] for r in g)]
        print(f"{mf:<26}" + "".join(f"  {v:>11}" for v in vals))

    # ---- what still needs doing
    def show(title, sel, hint=""):
        g = [r for r in rows if sel(r)]
        if not g:
            return
        print(f"\n{title} ({len(g)}){'  ->  ' + hint if hint else ''}")
        for r in g:
            extra = []
            if r["claimed"]:
                extra.append(f"claimed {r['claimed']}")
            if r["last_ckpt_epoch"] != "" and r["status"] != "trained":
                extra.append(f"last checkpoint ep={r['last_ckpt_epoch']}")
            print(f"  {r['run_id']:<62} {r['manifest']:<26} {'; '.join(extra)}")

    show("RUNNING NOW", lambda r: r["status"] == "running")
    show("FAILED OR INTERRUPTED: claimed, no epoch-99 checkpoint, not running", lambda r: r["status"] == "failed",
         "check tmux ls first; to retry: rmdir study/claims/<run_id>, then start the manifest's launcher in tmux")
    show("NOT STARTED", lambda r: r["status"] == "not started")
    show("DIVERGED (non-finite features)", lambda r: r["status"] == "diverged")
    show("TRAINED, NOT SCORED ON VALIDATION", lambda r: r["status"] == "trained" and not r["val_scored"],
         "python study_tools/eval_run.py --manifest <manifest>   (scores every finished, unscored run)")
    show("IN A TEST PLAN AND TRAINED, NOT TEST-SCORED", lambda r: r["status"] == "trained" and r["in_test_plan"] and not r["test_scored"],
         "python study_tools/eval_test.py --plan <its test plan>")
    show("IN A TEST PLAN, NOT TRAINED YET", lambda r: r["in_test_plan"] and r["status"] != "trained")
    show("TEST-SCORED, NO TEST MS", lambda r: r["test_scored"] and not r["test_ms"],
         "python study_tools/eval_test_ms.py --plan <its test plan>")
    if a.all:
        print(f"\nALL RUNS ({len(rows)})")
        for r in rows:
            flags = [k for k in ("val_scored", "in_test_plan", "test_scored", "test_ms") if r[k]]
            print(f"  {r['run_id']:<62} {r['status']:<12} {', '.join(flags)}")

    out = Path(a.csv)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run_id"])
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} runs; {len(alive)} training process(es) alive. Full table: {out}")


if __name__ == "__main__":
    main()
