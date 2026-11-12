"""Five strength levels per augmentation, shared by gallery.py and label_test.py.

speckle, contrast, blur : values chosen here (the training code takes a number directly).
motion, pixelate, jpeg  : MedMNIST-C severity levels 1..5 (index 0..4), via medmnistc_transforms.
The training blur is random: sigma ~ U[0.1, 2.0] per image, at the same probability as the others.
"""
import random
from PIL import Image, ImageFilter

import os, sys
sys.path.insert(0, os.getcwd())          # run the tools from the repository root
try:
    from MedMNIST.medmnistc_transforms import (MedMNISTCAugmentation, SpeckleStd, ContrastFactor, GaussianRange,
                                               JpegQuality, PixelateFactor, MotionBlurCustom)
except ImportError:                       # flat layout (used for testing outside the repository)
    from medmnistc_transforms import (MedMNISTCAugmentation, SpeckleStd, ContrastFactor, GaussianRange,
                                      JpegQuality, PixelateFactor, MotionBlurCustom)


class GaussianSigma:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = float(lo), float(lo if hi is None else hi)

    def __call__(self, img):
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(random.uniform(self.lo, self.hi)))

    def __repr__(self):
        return f"GaussianSigma({self.lo}, {self.hi})"


POOL = ["speckle", "contrast", "blur", "motion", "pixelate", "jpeg"]

SPECKLE_STD = [0.05, 0.10, 0.15, 0.20, 0.30]
CONTRAST_FACTOR = [0.9, 0.8, 0.7, 0.6, 0.5]
BLUR_SIGMA = [0.5, 1.0, 1.5, 2.0, 2.5]


def ladder(name, dataset="octmnist"):
    """List of (label, callable) for levels 1..5 of one augmentation."""
    if name == "speckle":
        return [(f"std {s}", SpeckleStd(s)) for s in SPECKLE_STD]
    if name == "contrast":
        return [(f"factor {f}", ContrastFactor(f)) for f in CONTRAST_FACTOR]
    if name == "blur":
        return [(f"sigma {s}", GaussianSigma(s)) for s in BLUR_SIGMA]
    if name in ("motion", "pixelate", "jpeg"):
        mc = {"motion": "motion_blur", "pixelate": "pixelate", "jpeg": "jpeg_compression"}[name]
        return [(f"L{i + 1} {MedMNISTCAugmentation(mc, i, dataset).param()}", MedMNISTCAugmentation(mc, i, dataset))
                for i in range(5)]
    raise KeyError(name)


# Gentler ladders for 64 px OCT, used after the first label test showed every MedMNIST-C level of motion blur,
# pixelate and JPEG (and Gaussian blur above sigma 0.5) cost more than the tolerance. Not MedMNIST-C values.
MILD = ["blur", "motion", "pixelate", "jpeg"]


def mild_ladder(name):
    if name == "blur":      # sigma ~ U[0.1, max]
        return [(f"sigma U[0.1,{m}]", GaussianRange(0.1, m)) for m in (0.6, 0.8, 1.0, 1.2, 1.5)]
    if name == "motion":    # radius, sigma, angle limited to +/-15 degrees
        return [(f"r{r} s{sg} +/-15deg", MotionBlurCustom(r, sg, 15)) for r, sg in ((3, 1), (4, 1.5), (5, 2), (6, 2.5), (8, 3))]
    if name == "pixelate":
        return [(f"factor {f}", PixelateFactor(f)) for f in (0.9, 0.8, 0.7, 0.6, 0.5)]
    if name == "jpeg":
        return [(f"quality {q}", JpegQuality(q)) for q in (90, 80, 70, 60, 50)]
    raise KeyError(name)


def training_blur():
    """The blur exactly as the current training code draws it."""
    return ("train range sigma U[0.1,2.0]", GaussianSigma(0.1, 2.0))


def from_levels(name, levels, dataset="octmnist"):
    """The callable for one augmentation at the strength written in levels.json."""
    lv = levels[name]
    if name == "speckle":
        return SpeckleStd(lv["std"])
    if name == "contrast":
        return ContrastFactor(lv["factor"])
    if name == "blur":
        return GaussianRange(0.1, lv.get("sigma_max", 2.0))
    if name == "motion" and "radius" in lv:
        return MotionBlurCustom(lv["radius"], lv["sigma"], lv.get("angle", 15))
    if name == "pixelate" and "factor" in lv:
        return PixelateFactor(lv["factor"])
    if name == "jpeg" and "quality" in lv:
        return JpegQuality(lv["quality"])
    mc = {"motion": "motion_blur", "pixelate": "pixelate", "jpeg": "jpeg_compression"}[name]
    return MedMNISTCAugmentation(mc, lv["level"], dataset)


# Extended ladders (29 Oct): where does each augmentation of the focused design start removing class information?
# contrast_wide continues the contrast ladder below 0.5; the others are new. Each level is applied as pretraining
# would apply it: a random crop of at least that area, a random rotation within +/- that angle, and so on.
EXTENDED = ["contrast_wide", "brightness", "rotation", "crop", "elastic"]


class CropArea:
    """Random resized crop keeping at least `area` of the image, back to the original size (as pretraining)."""

    def __init__(self, area):
        self.area = float(area)

    def __call__(self, img):
        from torchvision import transforms
        return transforms.RandomResizedCrop((img.height, img.width), scale=(self.area, 1.0), ratio=(3 / 4, 4 / 3),
                                            interpolation=transforms.InterpolationMode.BICUBIC)(img)

    def __repr__(self):
        return f"CropArea({self.area})"


class Rotate:
    """Random rotation within +/- deg degrees, black corners (the OCT background)."""

    def __init__(self, deg):
        self.deg = float(deg)

    def __call__(self, img):
        from torchvision import transforms
        return transforms.RandomRotation(self.deg, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)(img)

    def __repr__(self):
        return f"Rotate({self.deg})"


class Brightness:
    """Brightness factor drawn from [1 - s, 1 + s] (torchvision ColorJitter, as the stem's brightness jitter)."""

    def __init__(self, s):
        self.s = float(s)

    def __call__(self, img):
        from torchvision import transforms
        return transforms.ColorJitter(brightness=self.s)(img)

    def __repr__(self):
        return f"Brightness({self.s})"


class Elastic:
    """torchvision ElasticTransform with displacement strength alpha (sigma 5 px smoothing)."""

    def __init__(self, alpha, sigma=5.0):
        self.alpha, self.sigma = float(alpha), float(sigma)

    def __call__(self, img):
        from torchvision import transforms
        return transforms.ElasticTransform(alpha=self.alpha, sigma=self.sigma, fill=0)(img)

    def __repr__(self):
        return f"Elastic({self.alpha}, {self.sigma})"


def extended_ladder(name):
    if name == "contrast_wide":
        return [(f"factor {f}", ContrastFactor(f)) for f in (0.5, 0.4, 0.3, 0.2, 0.1)]
    if name == "brightness":
        return [(f"jitter {s}", Brightness(s)) for s in (0.2, 0.4, 0.6, 0.8, 0.95)]
    if name == "rotation":
        return [(f"+/-{d} deg", Rotate(d)) for d in (5, 10, 15, 20, 30)]
    if name == "crop":
        return [(f"min area {a}", CropArea(a)) for a in (0.5, 0.3, 0.2, 0.1, 0.08)]
    if name == "elastic":
        return [(f"alpha {al}", Elastic(al)) for al in (10, 25, 50, 75, 100)]
    raise KeyError(name)

