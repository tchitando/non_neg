"""PathMNIST augmentation pool: operations, strength ladders and the rule that turns label-test results into levels.

Pool (bit order = order of application in MedMNIST/pretrain_dataloader.py):
    contrast  contrast reduction toward the image mean (same formula as OCT), factor f
    rotflip   random multiple of 90 degrees + vertical flip 0.5 (one setting)
    stain     H&E stain jitter in optical density (Tellez et al. 2019), sigma
    defocus   disk-kernel defocus blur (MedMNIST-C kernel), radius in pixels
    pixelate  down- and up-sampling by a factor (as OCT)
    jpeg      JPEG at a quality (as OCT)
Every operation is applied with probability 0.5 per view, after the stem (crop 0.08, brightness jitter, horizontal flip).

Level rule (fixed before the label test is run):
    contrast, pixelate, jpeg  keep the OCT strength (0.8, 0.8, 70) when it passes, so the three shared operations have
                              the same setting on every dataset; otherwise the strongest passing level.
    stain, defocus            the strongest passing level.
    rotflip                   on if it passes (it only permutes pixels, so it should).
"""
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from MedMNIST.medmnistc_transforms import ContrastFactor, PixelateFactor, JpegQuality, RotFlip, StainJitterHED, DefocusRadius
except ImportError:
    from medmnistc_transforms import ContrastFactor, PixelateFactor, JpegQuality, RotFlip, StainJitterHED, DefocusRadius

POOL = ["contrast", "rotflip", "stain", "defocus", "pixelate", "jpeg"]
PROB = 0.5

LADDER_VALUES = {
    "contrast": [0.9, 0.8, 0.7, 0.6, 0.5],
    "rotflip": [1],
    "stain": [0.02, 0.05, 0.1, 0.15, 0.2],
    "defocus": [1.0, 1.5, 2.0, 2.5, 3.0],
    "pixelate": [0.9, 0.8, 0.7, 0.6, 0.5],
    "jpeg": [90, 80, 70, 60, 50],
}
SHARED_TARGET = {"contrast": 0.8, "pixelate": 0.8, "jpeg": 70}


def op(name, v):
    if name == "contrast":
        return ContrastFactor(v)
    if name == "rotflip":
        return RotFlip()
    if name == "stain":
        return StainJitterHED(v)
    if name == "defocus":
        return DefocusRadius(v, 0.1)
    if name == "pixelate":
        return PixelateFactor(v)
    if name == "jpeg":
        return JpegQuality(v)
    raise KeyError(name)


def label(name, v):
    return {"contrast": f"factor {v}", "rotflip": "rot90 + vflip", "stain": f"sigma {v}", "defocus": f"radius {v}",
            "pixelate": f"factor {v}", "jpeg": f"quality {v}"}[name]


def ladder(name):
    """List of (label, value, callable) for one operation."""
    return [(label(name, v), v, op(name, v)) for v in LADDER_VALUES[name]]


def choose_level(name, passing_values):
    """Apply the level rule to the values that passed. Returns the chosen value or None."""
    if not passing_values:
        return None
    if name in SHARED_TARGET and SHARED_TARGET[name] in passing_values:
        return SHARED_TARGET[name]
    vals = LADDER_VALUES[name]
    return max(passing_values, key=vals.index)        # ladders run from mild to strong


def yaml_block(name, v, on):
    """The augmentation-yaml lines for one operation (prob 0 when off)."""
    p = PROB if on else 0.0
    if name == "contrast":
        return f"  contrast_decrease:\n    prob: {p}\n    factor: {v}\n"
    if name == "rotflip":
        return f"  rot_flip:\n    prob: {p}\n"
    if name == "stain":
        return f"  stain_jitter:\n    prob: {p}\n    sigma: {v}\n"
    if name == "defocus":
        return f"  defocus_blur:\n    prob: {p}\n    radius: {v}\n    alias_blur: 0.1\n"
    if name == "pixelate":
        return f"  pixelate:\n    prob: {p}\n    factor: {v}\n"
    if name == "jpeg":
        return f"  jpeg_compression:\n    prob: {p}\n    quality: {v}\n"
    raise KeyError(name)
