#!/usr/bin/env python3
"""Part 8 (OCT, ResNet-50): single-augmentation strength curves on the crop-only base.

    python study_tools/make_croponly_strength_runs.py --dry-run
    python study_tools/make_croponly_strength_runs.py                    # the scan: seed 42 only (16 runs)
    python study_tools/make_croponly_strength_runs.py --seeds 43 44      # later: more seeds, for the levels that matter
    python study_tools/make_croponly_strength_runs.py --seeds 43 44 --only contrast_f050 blur_sig120
    python study_tools/make_croponly_strength_runs.py --seeds 45 46 --only croponly contrast_f050   # 5 seeds for the headline
    python study_tools/make_croponly_strength_runs.py --seeds 42 43 44 --only bright_b020 bright_b040 bright_b095
    python study_tools/make_croponly_strength_runs.py --only speckle_s005 speckle speckle_s020 motion_r3 motion motion_r8 \
        pixelate_f090 pixelate pixelate_f060 jpeg_q90 jpeg jpeg_q40      # the other four pool augmentations, 3 levels, seed 42

The other four augmentations of the OCT pool (speckle, motion blur, pixelation, JPEG; added 3 Nov 2026) at three
strengths each on the same base: milder than the screen level, the screen level (study/levels.json: the strongest
level that cost at most 2 points of balanced accuracy in the label tests), and stronger:
    speckle std            0.05      0.1 (screen)      0.2
    motion radius, sigma   3, 1      5, 2 (screen)     8, 3      (angle +/-15 degrees throughout)
    pixelate factor        0.9       0.8 (screen)      0.6       (lower = coarser)
    JPEG quality           90        70 (screen)       40        (lower = stronger compression)

Base: crop and resize only (random resized crop, minimum area 0.08, back to 64 px; brightness jitter and flip off),
the base of Chen et al. (2020, Fig. 5); trained in Part 7 as `croponly`. Each policy below adds ONE augmentation at
ONE strength to that base, so each level is compared with the base directly (no averaging over other augmentations).

    contrast factor   0.5 (new)   0.3 (new)                         0.1 (new; near-flat image, expected to damage)
                      the screen's contrast operation (a view's contrast lowered to exactly the factor, p 0.5);
                      with the base (1.0) and Part 7's croponly_contrast (0.8) the curve has five points spread
                      from no contrast change to near-flat, so the best level is bracketed by worse ones
    blur sigma max    0.6 (new)   1.2 (new)                         2.0 (new; SimCLR's range 0.1-2.0)
    crop min area     0.5 (new)   0.2 (new)                         0.08 (the base itself)

First a one-seed scan: seed 42 x NCL and SimCLR = 16 runs. Seeds 43 and 44 are added afterwards with --seeds, for
all levels or only the ones chosen with --only; each call writes its own manifest and launcher
(study/manifest_part8.csv for seed 42, study/manifest_part8_s43-44.csv for --seeds 43 44, ...). Augmentation files are built from p_contrast / p_blur / p_stem with
the flip probability set to 0 and colour jitter off (as in Part 7) and the one strength value changed; run files are
copies of the contrast / blur / stem run files with only the augmentation file, name and seed changed.
Outputs: study/manifest_part8.csv, study/launch_part8.sh and the yamls. Existing files with different content are
never overwritten. The old-stem strength runs (Part 6, make_strength_runs.py) are replaced by this design.
"""
import argparse, csv, json, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
SET = {   # what to change in the source augmentation file: (pattern, levels.json key)
    "contrast": (r"(contrast_decrease:\s*\n\s*prob:\s*\S+\s*\n\s*factor:\s*)\S+", "factor"),
    "blur": (r"(blur_custom:\s*\n\s*prob:\s*\S+\s*\n\s*sigma_lo:\s*\S+\s*\n\s*sigma_hi:\s*)\S+", "sigma_max"),
    "crop": (r"(crop_min_scale:\s*)\S+", None),
    "bright": (r"(color_jitter:\s*\n\s*prob:\s*\S+\s*\n\s*brightness:\s*)\S+", "s"),
    "speckle": (r"(speckle_noise:\s*\n\s*prob:\s*\S+\s*\n\s*std:\s*)\S+", "std"),
    "motion": (r"(motion_blur:\s*\n\s*prob:\s*\S+\s*\n\s*radius:\s*)\S+(\s*\n\s*sigma:\s*)\S+", None),
    "pixelate": (r"(pixelate:\s*\n\s*prob:\s*\S+\s*\n\s*factor:\s*)\S+", "factor"),
    "jpeg": (r"(jpeg_compression:\s*\n\s*prob:\s*\S+\s*\n\s*quality:\s*)\S+", "quality"),
    "cj": (r"(color_jitter:\s*\n\s*prob:\s*)\S+(\s*\n\s*brightness:\s*)\S+(\s*\n\s*contrast:\s*)\S+", "s"),
}
# (new policy id, source policy, which value, level)
POLICIES = [
    ("croponly", "stem", None, None),          # the base itself (Part 7), for extra seeds: --only croponly
    ("croponly_contrast", "contrast", None, None),   # contrast 0.8 (Part 7), for extra seeds: --only contrast
    ("croponly_contrast_f050", "contrast", "contrast", 0.5),
    ("croponly_contrast_f030", "contrast", "contrast", 0.3),
    ("croponly_contrast_f010", "contrast", "contrast", 0.1),
    ("croponly_crop050", "stem", "crop", 0.5),
    ("croponly_crop020", "stem", "crop", 0.2),
    ("croponly_blur_sig060", "blur", "blur", 0.6),
    ("croponly_blur", "blur", "blur", 1.2),
    # brightness jitter on crop only (the stem's colour jitter, p 0.8, flip off), strength s: factor in [1-s, 1+s];
    # s 0.8 is Part 9's croponly_bright. Not part of the 16-run scan; ask for them with --only.
    ("croponly_bright", "stem", "bright", 0.8),
    ("croponly_bright_b020", "stem", "bright", 0.2),
    ("croponly_bright_b040", "stem", "bright", 0.4),
    ("croponly_bright_b095", "stem", "bright", 0.95),
    ("croponly_blur_sig200", "blur", "blur", 2.0),
    # the other four pool augmentations (3 Nov 2026): mild, screen level, strong; ask for them with --only
    ("croponly_speckle_s005", "speckle", "speckle", 0.05),
    ("croponly_speckle", "speckle", "speckle", 0.1),
    ("croponly_speckle_s020", "speckle", "speckle", 0.2),
    ("croponly_motion_r3", "motion", "motion", (3, 1)),
    ("croponly_motion", "motion", "motion", (5, 2)),
    ("croponly_motion_r8", "motion", "motion", (8, 3)),
    ("croponly_pixelate_f090", "pixelate", "pixelate", 0.9),
    ("croponly_pixelate", "pixelate", "pixelate", 0.8),
    ("croponly_pixelate_f060", "pixelate", "pixelate", 0.6),
    ("croponly_jpeg_q90", "jpeg", "jpeg", 90),
    ("croponly_jpeg", "jpeg", "jpeg", 70),
    ("croponly_jpeg_q40", "jpeg", "jpeg", 40),
]
NEW_POOL = {"speckle", "motion", "pixelate", "jpeg"}


def sub_once(pat, repl, text, what, flags=re.M):
    new, n = re.subn(pat, repl, text, flags=flags)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--only", nargs="*", default=None, help="policy ids without the 'croponly_' prefix, e.g. contrast_f050")
    a = ap.parse_args()
    tag = "" if a.seeds == [42] else "_s" + "-".join(str(x) for x in a.seeds)
    mans = {f: [r["run_id"] for r in csv.DictReader(open(f))] for f in sorted(Path("study").glob("manifest_part*.csv"))}
    src_rows = {r["run_id"]: r for f in mans for r in csv.DictReader(open(f))}
    pols = [p for p in POLICIES if (a.only is None and p[2] not in (None, "bright") and p[2] not in NEW_POOL) or
            (a.only is not None and (p[0] if p[0] == "croponly" else p[0].replace("croponly_", "")) in a.only)]
    if not pols:
        raise SystemExit("no policy matches --only")
    files, rows = {}, []
    for seed in a.seeds:
        for pid, pol, what, val in pols:
            aug_file = f"p_{pid}.yaml"
            t = (OCT / "augmentations" / f"p_{pol}.yaml").read_text()
            t = sub_once(r"(horizontal_flip:\s*\n\s*prob:\s*)\S+", r"\g<1>0.0", t, "horizontal_flip prob", flags=0)
            if what is None:      # the Part 7 policies: colour jitter off, nothing else changed
                t = sub_once(r"(color_jitter:\s*\n\s*prob:\s*)\S+", r"\g<1>0.0", t, "color_jitter prob", flags=0)
            elif what == "cj":
                t = sub_once(SET["cj"][0], rf"\g<1>0.5\g<2>0.0\g<3>{val}", t, "color_jitter block", flags=0)
            elif what == "bright":   # keep the stem's colour jitter (p 0.8), change only its brightness strength
                t = sub_once(SET["bright"][0], rf"\g<1>{val}", t, "color_jitter brightness", flags=0)
            elif what == "motion":
                t = sub_once(r"(color_jitter:\s*\n\s*prob:\s*)\S+", r"\g<1>0.0", t, "color_jitter prob", flags=0)
                t = sub_once(SET["motion"][0], rf"\g<1>{val[0]}\g<2>{val[1]}", t, "motion radius and sigma", flags=0)
            else:
                t = sub_once(r"(color_jitter:\s*\n\s*prob:\s*)\S+", r"\g<1>0.0", t, "color_jitter prob", flags=0)
                t = sub_once(SET[what][0], rf"\g<1>{val}", t, f"{what} value", flags=0)
            files[OCT / "augmentations" / aug_file] = t
            for obj in OBJS:
                base_id = f"{obj}-resnet50-octmnist-{pol}"
                if base_id not in src_rows:
                    raise SystemExit(f"{base_id} not in the earlier manifests")
                src = src_rows[base_id]
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                r = (OCT / f"{obj}_rn50_p_{pol}.yaml").read_text()
                r = sub_once(r"^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", r, "augmentations")
                r = sub_once(r'^name:\s*".*"\s*$', f'name: "{run_id}"', r, "name")
                r = sub_once(r"^seed:\s*\d+\s*$", f"seed: {seed}", r, "seed")
                files[OCT / f"{cfg}.yaml"] = r
                levels, crop = src["levels"], src["crop_min_scale"]
                if what is None:
                    pass
                elif what == "crop":
                    crop = val
                elif what == "cj":
                    lv = json.loads(levels); lv["contrast_jitter"] = {"s": val, "prob": 0.5}; levels = json.dumps(lv)
                elif what == "bright":
                    lv = json.loads(levels or "{}"); lv["brightness"] = {"s": val}; levels = json.dumps(lv)
                elif what == "motion":
                    lv = json.loads(levels); lv.setdefault("motion", {}).update(radius=val[0], sigma=val[1]); levels = json.dumps(lv)
                else:
                    lv = json.loads(levels or "{}"); lv.setdefault(pol, {})[SET[what][1]] = val; levels = json.dumps(lv)
                rows.append(dict(run_id=run_id, part=8, objective=obj, seed=seed, policy_id=pid, bitmask=src["bitmask"],
                                 augs_on=src["augs_on"], n_augs=src["n_augs"], prob=src["prob"], levels=levels,
                                 crop_min_scale=crop, aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg,
                                 priority=1, kind="croponly_strength"))
    # a manifest that already holds exactly these runs is rewritten (same content); any other overlap is refused;
    # otherwise the first free name is used (manifest_part8_s43-44.csv, then ..._s43-44b.csv, ...), never overwriting
    new_ids = [r["run_id"] for r in rows]
    same = [f for f, ids in mans.items() if sorted(ids) == sorted(new_ids)]
    if same:
        man = same[0]
        tag = man.stem[len("manifest_part8"):]
    else:
        dup = sorted(i for i in new_ids if i in src_rows)
        if dup:
            raise SystemExit(f"run ids already planned in another manifest: {dup}")
        for suffix in [""] + list("bcdefghij"):
            man = Path(f"study/manifest_part8{tag}{suffix}.csv")
            if not man.exists():
                tag = tag + suffix
                break
    for r in rows:
        print(f"P{r['priority']}  {r['run_id']:52s} seed {r['seed']}  {r['levels']:32s} crop {r['crop_min_scale']}")
    print(f"\n{len(rows)} runs, {len(files)} new files")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path(f"study/launch_part8{tag}.sh")
    sh.write_text(LAUNCH.replace("study/manifest_part8.csv", str(man)).replace("launch_part8.sh", sh.name))
    sh.chmod(0o755)
    print(f"wrote {man}, {sh} and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 8 launcher (crop-only strength curves). Several copies can run at once: each run is claimed with an atomic mkdir in
# study/claims/, and a run counts as done only when its epoch-99 checkpoint exists. Runs start in priority order.
#   CUDA_VISIBLE_DEVICES=1 bash study/launch_part8.sh
#   CUDA_VISIBLE_DEVICES=0 bash study/launch_part8.sh '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part8.txt; TIMING=study/timing_part8.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part8_queue_$$.txt
import csv
for r in csv.DictReader(open("study/manifest_part8.csv")):
    print(r["run_id"] + "," + r["launch_cfg"])
PY
while IFS=, read -r RUN CFG; do
  echo "$RUN" | grep -Eq "$PAT" || continue
  grep -qx "$RUN" "$DONE" && continue
  mkdir "$CLAIMS/$RUN" 2>/dev/null || continue
  echo "=== $RUN  (GPU ${CUDA_VISIBLE_DEVICES:-?})"; T0=$(date +%s)
  python3 MedMNIST/main_pretrain.py --config-path MedMNIST/scripts/pretrain/octmnist --config-name "$CFG"
  if find trained_models -name "$RUN-offline-*-ep=99.ckpt" 2>/dev/null | grep -q .; then
    S=$(( $(date +%s) - T0 )); echo "$RUN" >> "$DONE"
    echo "$RUN,$S,$(python3 -c "print(round($S/3600,3))"),${CUDA_VISIBLE_DEVICES:-?}" >> "$TIMING"
  else
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part8.txt
  fi
done < /tmp/part8_queue_$$.txt
rm -f /tmp/part8_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
