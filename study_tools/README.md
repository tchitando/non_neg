# Part 1: augmentation study on OCT. What is here and what to do

## What the audit found (read this before trusting any number)
- **Stem (A0)**: random resized crop min scale 0.08, horizontal flip 0.5, brightness jitter (p 0.8, 0.8). Contrast, saturation, hue 0. No blur, no speckle, no contrast decrease. `make_runs.py` regenerates it and it matches A0.yaml key for key (the only additions are three new blocks with prob 0).
- **Recipe** (never changed): ResNet-50, LARS lr 0.4, batch 256, 100 epochs, temperature 0.2, projector 2048 to 256, NCL = `non_neg: rep_relu`, seed 42 unless a stem seed run.
- **Order in the code**: crop, brightness, then speckle, contrast, and the four lossy augmentations (Gaussian blur through `blur_custom`, motion blur, pixelate, JPEG), then ToTensor and Normalize. Bit order in tables is speckle, contrast, blur, motion, pixelate, jpeg, which is also the order of application.
- **Probabilities**: every pool augmentation at p 0.5 (same as A6), set in `study/levels.json`.
- **Strengths (final, `study/levels.json`, p = 0.5 each)**: speckle std 0.1, contrast factor 0.8, Gaussian blur sigma uniform in [0.1, 1.2], motion blur radius 5 / sigma 2 / angle 15, pixelate factor 0.8, JPEG quality 70. The speckle/contrast values are the A6 values. Blur, motion, pixelate and JPEG are custom levels chosen with the label-preservation test (`label_test_aug.py --mild`); they are NOT MedMNIST-C levels, because the MedMNIST-C ladders for motion, pixelate and JPEG fail the label test at every level at 64 px. State this in the write-up. Speckle noise is drawn independently per RGB channel (coloured speckle on grayscale scans), kept for comparability with the earlier A6/E1 runs; state this too.
- **CIFAR-derived policy** = `augmentations/symmetric.yaml` (used by `ncl_cifar_augs.yaml` / `simclr_cifar_augs.yaml`): crop 0.08, colour jitter 0.8 with brightness, contrast, saturation 0.8 and hue 0.2, grayscale 0.2, Gaussian blur 0.5, flip 0.5. `make_runs.py --cifar-yaml MedMNIST/scripts/pretrain/octmnist/augmentations/symmetric.yaml` copies it as `p_cifar` and adds the three new blocks at prob 0. Note that this reference differs from the stem in more than one way (colour channels on grayscale images, grayscale 0.2, blur).
- **medmnistc 0.1.0 problems** (handled in `medmnistc_transforms.py`): motion blur returns a broken array for grayscale images that are not 224 px; it uses `np.fromstring` (fails on numpy 2); impulse noise reuses one fixed noise pattern (not used here). The pixelate ladder is relative (fractions of the image), not pixel-based; the issue at 64 px is that little detail is available to lose, so the gentler custom levels above are used.

## Files (put them as shown, run everything from the repository root)
| File | Where | What |
|---|---|---|
| `medmnistc_transforms.py` | `MedMNIST/` | fixed MedMNIST-C wrapper plus `build_extra_augmentations` |
| `pretrain_dataloader.patch` | apply with `patch -p0 MedMNIST/pretrain_dataloader.py < ...` | adds the three new augmentations after speckle/contrast; imports medmnistc only when one is switched on |
| `aug_ladders.py, gallery.py, label_test.py, make_runs.py, test_transforms.py` | `study_tools/` | the tools below |

`pip install medmnistc` (needs ImageMagick/wand and opencv; your `ml` env already has them for E6).

## Order of work
1. `python study_tools/test_transforms.py` : 15 combinations, shapes and timing. Passed here on a synthetic image (motion blur about 8 ms per 64 px image, the others under 1 ms).
2. `python study_tools/make_runs.py --part 1 --cifar-yaml <file>` : writes the yamls, `study/manifest_part1.csv` and `study/launch_part1.sh`. **Check `--launch-cmd` against how you start runs** (default: `python3 MedMNIST/main_pretrain.py --config-path ... --config-name ...`).
3. Start the stem NCL run alone. `study/timing_part1.csv` then gives **H**, the GPU-hours per run, which decides whether all 64 combinations fit for both objectives. The stem does not depend on the levels.
4. While it runs: `python study_tools/gallery.py --npz MedMNIST/octmnist_64.npz` and look at `study/gallery/*.png`, then `python study_tools/label_test.py --npz MedMNIST/octmnist_64.npz`. Write the chosen levels into `study/levels.json` and rerun `make_runs.py` (the stem and CIFAR yamls do not change).
5. Launch the rest of Part 1: `bash study/launch_part1.sh` (skips finished runs).

Part 1 = stem x seeds 42/43/44 x {NCL, SimCLR} (6) + six singles x 2 (12) + CIFAR x 2 = 20 runs. `--part 2` gives 44 runs, `--part 3` the 70 triple/four-way runs (the crop sweep is not generated yet).

## Decision after Part 1 (validation only, written down before Part 2)
Is any single effect larger than the three-seed stem spread? Are NCL and SimCLR rankings consistent? Does any level fail the label test? Keep the two rankings (balanced accuracy and MS/SC*) separate. SC* can only rank NCL runs: SimCLR's SC* is at the class-prior floor by construction, so SimCLR is compared on MS and the probe.

## Tested vs not tested
- Tested here: transforms (shapes, levels, timing), run/yaml generator (stem identical to A0, dry runs for Parts 2 and 3: 22 and 35 policies), launch script syntax, dataloader patch applies and compiles, gallery on a synthetic image.
- **Not tested** (no torch, no OCT data, no GPU here): `label_test.py` (syntax-checked only), the patched dataloader inside a real run, `gallery.py` on real OCT data, the launch script on your server.
- **Not written yet**: the per-run evaluation that fills the master results table (validation balanced-accuracy probe, kNN, SC, SC*, MS, alive units, effective dimension, run ID, checkpoint path). It should reuse your `eval_linear_octmnist.py`, `eval_sc_balanced_all.py` and `Mono_octmnist.py`; note that `eval_linear_octmnist.py` reports plain accuracy, not validation balanced accuracy.
