#!/usr/bin/env python3
"""Part 12 (OCT, ResNet-50): pairs of augmentations on the crop-only base, for the combination heatmap.
Version 12 Nov b: adds --combo, a policy with any set of the six operations on the crop-only base, e.g. all six at
their swept levels: --combo bright+contrast+blur+flip+speckle+jpeg (the comparison with the SimCLR defaults).
Version of 12 Nov: adds speckle (std 0.1) and JPEG (quality 70), the levels the selection rule picked on 11 Nov,
--skip-planned, and seed-major ordering (every seed-42 run comes before any seed-43 run).

    python study_tools/make_croponly_pair_runs_12nov_b.py --blur 0.6 --combo bright+contrast+blur+flip+speckle+jpeg --seeds 42 43 44 --dry-run
                                                                      # the six swept operations together, seeds 42-44
    python study_tools/make_croponly_pair_runs_12nov_b.py --blur 0.6 --pairs all --seeds 42 43 44 --skip-planned --dry-run
                                                                      # every heatmap pair not yet planned, seeds 42-44
    python study_tools/make_croponly_pair_runs.py --blur 0.6 --pairs bright+blur contrast+blur blur+flip --dry-run
                                                                      # the blur pairs at seed 42: 3 pairs x 2 methods = 6 runs
    python study_tools/make_croponly_pair_runs.py --blur 0.6 --pairs bright+blur --seeds 43 44 --dry-run
                                                                      # later: seeds 43 and 44 for a cell the paper reports

Base: crop only (random resized crop, minimum area 0.08, back to 64 px, nothing else), as Parts 7 and 8. Each policy
turns on TWO augmentations, each at the strength chosen from its single-augmentation sweep on the same base (the level
that passes the label-preservation screen with the highest validation backbone balanced accuracy, averaged over NCL and
SimCLR and over seeds 42-44; the same level for both methods):
    brightness   colour jitter, brightness s 0.8, p 0.8        (the stem's setting)
    contrast     contrast decrease, factor 0.3, p 0.5
    flip         horizontal flip, p 0.5                         (the stem's setting)
    blur         Gaussian blur, sigma ~ U[0.1, s], p 0.5        (s given with --blur)
    speckle      speckle noise, std 0.1, p 0.5                  (as croponly_speckle)
    jpeg         JPEG compression, quality 70, p 0.5            (as croponly_jpeg)
brightness + flip is the stem (Part 1), already trained at seeds 42-44, so it is not made again.

Seeds: the heatmap is trained at ONE seed per cell (seed 42, the default), as in SimCLR's Fig. 5. Seeds 43 and 44 are
added afterwards with --seeds 43 44, only for the cells that are reported with a mean and SD. Brightness + contrast was
trained at seeds 42-44 before this change (study/manifest_part12.csv); contrast + flip seeds 43 and 44 were stopped.

Augmentation files are built from p_stem.yaml: every block is switched off except the two augmentations of the pair,
and the script checks that nothing else is on. Run files are copies of the stem run files with only the augmentation
file, name and seed changed. Run ids follow the existing pattern (seed 42 without suffix, -s43 / -s44 otherwise).
Outputs: study/manifest_part12.csv (then manifest_part12b.csv, ... ; never overwriting), a matching launcher
study/launch_part12*.sh and the yamls. Existing files with different content are never overwritten, and run ids that
are already planned in another manifest are refused.
"""
import argparse, csv, json, re
from pathlib import Path

OCT = Path("MedMNIST/scripts/pretrain/octmnist")
OBJS = ["ncl", "simclr"]
ORDER = ["bright", "contrast", "blur", "flip", "speckle", "jpeg"]   # order of the parts in a policy id
POOL = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]   # bitmask order used in every manifest
CONTRAST = 0.3
BRIGHT = 0.8
SPECKLE = 0.1
JPEG = 70
PAIRS = ["bright+contrast", "bright+blur", "contrast+blur", "contrast+flip", "blur+flip",       # bright+flip = the stem
         "bright+speckle", "contrast+speckle", "blur+speckle", "flip+speckle",
         "bright+jpeg", "contrast+jpeg", "blur+jpeg", "flip+jpeg", "speckle+jpeg"]
DEFAULT = ["bright+contrast", "contrast+flip"]


def sub_once(pat, repl, text, what):
    new, n = re.subn(pat, repl, text)
    if n != 1:
        raise SystemExit(f"expected exactly one '{what}' match, found {n}")
    return new


def value(text, block, key):
    m = re.search(rf"{block}:\s*\n(?:\s*\w+:.*\n)*?\s*{key}:\s*(\S+)", text)
    if not m:
        raise SystemExit(f"{block}.{key} not found")
    return float(m.group(1))


def part_id(t, blur):
    if t == "blur":
        return f"blur_sig{round(blur * 100):03d}"
    return {"bright": "bright", "contrast": "contrast_f030", "flip": "flip",
            "speckle": f"speckle_s{round(SPECKLE * 100):03d}", "jpeg": f"jpeg_q{JPEG}"}[t]


def aug_text(src, on, blur):
    t = src
    t = sub_once(r"(color_jitter:\s*\n\s*prob:\s*)\S+", r"\g<1>" + ("0.8" if "bright" in on else "0.0"), t, "color_jitter prob")
    t = sub_once(r"(horizontal_flip:\s*\n\s*prob:\s*)\S+", r"\g<1>" + ("0.5" if "flip" in on else "0.0"), t, "flip prob")
    t = sub_once(r"(contrast_decrease:\s*\n\s*prob:\s*)\S+(\s*\n\s*factor:\s*)\S+",
                 rf"\g<1>0.5\g<2>{CONTRAST}" if "contrast" in on else r"\g<1>0.0\g<2>0.8", t, "contrast_decrease block")
    t = sub_once(r"(blur_custom:\s*\n\s*prob:\s*)\S+(\s*\n\s*sigma_lo:\s*\S+\s*\n\s*sigma_hi:\s*)\S+",
                 rf"\g<1>0.5\g<2>{blur}" if "blur" in on else r"\g<1>0.0\g<2>1.2", t, "blur_custom block")
    t = sub_once(r"(speckle_noise:\s*\n\s*prob:\s*)\S+(\s*\n\s*std:\s*)\S+",
                 rf"\g<1>0.5\g<2>{SPECKLE}" if "speckle" in on else r"\g<1>0.0\g<2>0.1", t, "speckle_noise block")
    t = sub_once(r"(jpeg_compression:\s*\n\s*prob:\s*)\S+(\s*\n\s*quality:\s*)\S+",
                 rf"\g<1>0.5\g<2>{JPEG}" if "jpeg" in on else r"\g<1>0.0\g<2>70", t, "jpeg_compression block")
    # checks: crop 0.08, brightness strength 0.8 and no other colour jitter, every other operation off
    assert value(t, "rrc", "crop_min_scale") == 0.08, "crop_min_scale is not 0.08"
    assert value(t, "color_jitter", "brightness") == BRIGHT, "brightness strength is not 0.8"
    for k in ("contrast", "saturation", "hue"):
        assert value(t, "color_jitter", k) == 0.0, f"color_jitter {k} is not 0"
    for blk in ("grayscale", "gaussian_blur", "solarization", "equalization", "motion_blur", "pixelate"):
        assert value(t, blk, "prob") == 0.0, f"{blk} is switched on in p_stem.yaml"
    on_now = {"bright": value(t, "color_jitter", "prob") > 0, "flip": value(t, "horizontal_flip", "prob") > 0,
              "contrast": value(t, "contrast_decrease", "prob") > 0, "blur": value(t, "blur_custom", "prob") > 0,
              "speckle": value(t, "speckle_noise", "prob") > 0, "jpeg": value(t, "jpeg_compression", "prob") > 0}
    if "speckle" in on:
        assert value(t, "speckle_noise", "std") == SPECKLE, "speckle std"
    if "jpeg" in on:
        assert value(t, "jpeg_compression", "quality") == JPEG, "jpeg quality"
    assert {k for k, v in on_now.items() if v} == set(on), f"switched on {on_now}, wanted {on}"
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="default: seed 42 only (one seed per heatmap cell); add 43 44 later for the cells that are reported")
    ap.add_argument("--pairs", nargs="+", default=DEFAULT, choices=PAIRS + ["all"])
    ap.add_argument("--combo", nargs="+", default=None,
                    help="policies with any set of operations joined by '+', e.g. bright+contrast+blur+flip+speckle+jpeg; "
                         "replaces --pairs")
    ap.add_argument("--skip-planned", action="store_true",
                    help="leave out run ids already in a study/manifest_part*.csv (listed) instead of stopping")
    ap.add_argument("--blur", type=float, default=None, help="blur sigma max for the pairs with blur (e.g. 0.6 or 1.2)")
    a = ap.parse_args()
    if a.combo:
        for c in a.combo:
            parts = c.split("+")
            if len(parts) < 2 or len(set(parts)) != len(parts) or not set(parts) <= set(ORDER):
                raise SystemExit(f"--combo {c}: use two or more distinct operations from {ORDER}")
            if set(parts) == {"bright", "flip"}:
                raise SystemExit("bright+flip is the stem (already trained)")
        a.pairs = a.combo
    elif "all" in a.pairs:
        a.pairs = PAIRS
    if any("blur" in p for p in a.pairs) and a.blur is None:
        raise SystemExit("pairs with blur need --blur (the chosen sigma max)")
    mans = {f: [r["run_id"] for r in csv.DictReader(open(f))] for f in sorted(Path("study").glob("manifest_part*.csv"))}
    planned = {i for ids in mans.values() for i in ids}
    stem_src = (OCT / "augmentations" / "p_stem.yaml").read_text()
    files, rows, skipped = {}, [], []
    for pair in a.pairs:
        on = sorted(pair.split("+"), key=ORDER.index)
        pid = "croponly_" + "_".join(part_id(t, a.blur) for t in on)
        aug_file = f"p_{pid}.yaml"
        files[OCT / "augmentations" / aug_file] = aug_text(stem_src, on, a.blur)
        lv = {}
        if "bright" in on: lv["brightness"] = {"s": BRIGHT}
        if "contrast" in on: lv["contrast"] = {"factor": CONTRAST}
        if "blur" in on: lv["blur"] = {"sigma_max": a.blur}
        if "flip" in on: lv["flip"] = {"prob": 0.5}
        if "speckle" in on: lv["speckle"] = {"std": SPECKLE}
        if "jpeg" in on: lv["jpeg"] = {"quality": JPEG}
        pool_on = [p for p in POOL if p in on]
        for seed in a.seeds:
            for obj in OBJS:
                run_id = f"{obj}-resnet50-octmnist-{pid}" + ("" if seed == 42 else f"-s{seed}")
                cfg = f"{obj}_rn50_p_{pid}" + ("" if seed == 42 else f"_s{seed}")
                if a.skip_planned and run_id in planned:
                    skipped.append(run_id)
                    continue
                r = (OCT / f"{obj}_rn50_p_stem.yaml").read_text()
                r = sub_once(r"(?m)^(\s*-\s*augmentations:\s*)\S+\s*$", rf"\g<1>{aug_file}", r, "augmentations")
                r = sub_once(r'(?m)^name:\s*".*"\s*$', f'name: "{run_id}"', r, "name")
                r = sub_once(r"(?m)^seed:\s*\d+\s*$", f"seed: {seed}", r, "seed")
                files[OCT / f"{cfg}.yaml"] = r
                rows.append(dict(run_id=run_id, part=12, objective=obj, seed=seed, policy_id=pid,
                                 bitmask="".join("1" if p in on else "0" for p in POOL), augs_on="+".join(pool_on),
                                 n_augs=len(pool_on), prob=0.5, levels=json.dumps(lv), crop_min_scale=0.08,
                                 aug_yaml=aug_file, run_yaml=f"{cfg}.yaml", launch_cfg=cfg, priority=1,
                                 kind="croponly_pair" if len(on) == 2 else "croponly_combo"))
    rows.sort(key=lambda r: (a.seeds.index(r["seed"]), a.pairs.index(next(p for p in a.pairs
              if "croponly_" + "_".join(part_id(t, a.blur) for t in sorted(p.split("+"), key=ORDER.index)) == r["policy_id"])),
              OBJS.index(r["objective"])))                                   # seed-major: all seed-42 runs first
    if skipped:
        print(f"skipped {len(skipped)} run ids already planned in a manifest:")
        for i in skipped:
            print("   ", i)
    if not rows:
        raise SystemExit("nothing new to plan")
    new_ids = [r["run_id"] for r in rows]
    same = [f for f, ids in mans.items() if sorted(ids) == sorted(new_ids)]
    if same:
        man = same[0]
    else:
        dup = sorted(i for i in new_ids if i in planned)
        if dup:
            raise SystemExit(f"run ids already planned in another manifest: {dup}")
        man = next(Path(f"study/manifest_part12{s}.csv") for s in [""] + list("bcdefghij")
                   if not Path(f"study/manifest_part12{s}.csv").exists())
    tag = man.stem[len("manifest_part12"):]
    for r in rows:
        print(f"{r['run_id']:58s} seed {r['seed']}  {r['levels']}")
    print(f"\n{len(rows)} runs, {len(files)} new files -> {man}, study/launch_part12{tag}.sh")
    if a.dry_run:
        print("dry run: nothing written"); return
    for p, t in files.items():
        if p.exists() and p.read_text() != t:
            raise SystemExit(f"{p} exists with different content; not overwriting")
    for p, t in files.items():
        p.write_text(t)
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sh = Path(f"study/launch_part12{tag}.sh")
    sh.write_text(LAUNCH.replace("@MANIFEST@", str(man)).replace("@LAUNCHER@", sh.name))
    sh.chmod(0o755)
    print(f"wrote {man}, {sh} and the yamls")


LAUNCH = r'''#!/usr/bin/env bash
# Part 12 launcher (crop-only pairs, @MANIFEST@). Several copies can run at once: each run is claimed with an atomic
# mkdir in study/claims/, and a run counts as done only when its epoch-99 checkpoint exists.
#   CUDA_VISIBLE_DEVICES=0 bash study/@LAUNCHER@
#   CUDA_VISIBLE_DEVICES=0 bash study/@LAUNCHER@ '^ncl-'    # optional regex filter on run ids
# If a run was interrupted, remove its claim to retry it:  rmdir study/claims/<run_id>
cd "$(dirname "$0")/.." || exit 1
DONE=study/done_part12.txt; TIMING=study/timing_part12.csv; CLAIMS=study/claims
mkdir -p "$CLAIMS"; touch "$DONE"; [ -f "$TIMING" ] || echo "run_id,seconds,gpu_hours,gpu" > "$TIMING"
PAT="${1:-.}"
python3 - <<'PY' > /tmp/part12_queue_$$.txt
import csv
for r in csv.DictReader(open("@MANIFEST@")):
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
    echo "NOT FINISHED $RUN (no epoch-99 checkpoint); claim kept, remove it to retry" | tee -a study/failed_part12.txt
  fi
done < /tmp/part12_queue_$$.txt
rm -f /tmp/part12_queue_$$.txt
echo "launcher finished"
'''

if __name__ == "__main__":
    main()
