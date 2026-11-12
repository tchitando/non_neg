"""MedMNIST-C corruptions as training augmentations for 64 px grayscale OCT.

Place this file next to MedMNIST/pretrain_dataloader.py (see dataloader_patch.md).
Requires the `medmnistc` package (pip install medmnistc).

Known problems in medmnistc 0.1.0 that this module works around:
  1. MotionBlur.apply() returns a (H, 3) array for grayscale images that are not 224x224,
     because of a hard-coded `img.shape != (224, 224)` check. `_motion_blur` below handles
     grayscale output of any size.
  2. MotionBlur uses `np.fromstring`, which fails on numpy >= 2. `_motion_blur` uses frombuffer.
  3. ImpulseNoise in augmentation mode uses a fixed-seed generator (the same noise pattern on
     every call) and in severity mode refers to a missing `self.rng`. It is not used here.
  4. Severity ladders for blur and pixelate are pixel-based and look designed for 224 px images.
     At 64 px the same values are relatively about 3.5x stronger. Choose levels with the
     image grid and the label-preservation test, not from the ladder alone.
"""
import ctypes, os, sys
# ImageMagick (needed by wand for motion blur) is installed inside the conda env but not on the library path.
# Load it by path first, as eval_e6_corruptions.py does. Harmless when it is already found.
for _lib in ("libMagickCore-7.Q16HDRI.so.10", "libMagickWand-7.Q16HDRI.so.10"):
    for _dir in (os.path.join(sys.prefix, "lib"), os.path.join(os.environ.get("CONDA_PREFIX", ""), "lib")):
        _p = os.path.join(_dir, _lib)
        if os.path.exists(_p):
            try:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break

import random
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from medmnistc.corruptions.registry import CORRUPTIONS_DS
from medmnistc.corruptions.filter import MotionImage


def _motion_blur(img: Image.Image, radius: float, sigma: float) -> np.ndarray:
    """ImageMagick motion blur as in medmnistc, with correct handling of grayscale output."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    m = MotionImage(blob=buf.getvalue())
    m.motion_blur(radius=radius, sigma=sigma, angle=np.random.uniform(-45, 45))
    out = cv2.imdecode(np.frombuffer(m.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
    buf.close()
    if out.ndim == 2:  # ImageMagick wrote a grayscale PNG
        out = np.stack([out, out, out], axis=-1)
    else:  # BGR(A) -> RGB
        out = out[..., [2, 1, 0]]
    return np.clip(out, 0, 255).astype(np.uint8)


class MedMNISTCAugmentation:
    """Apply one MedMNIST-C corruption at a fixed severity level (index 0..4 = levels 1..5).

    Input and output are PIL RGB images of the same size.
    """

    def __init__(self, name: str, level: int, dataset: str = "octmnist"):
        self.name = name
        self.level = int(level)
        self.dataset = dataset
        self.corruption = CORRUPTIONS_DS[dataset][name]
        n = len(self.corruption.severity_params)
        if not 0 <= self.level < n:
            raise ValueError(f"{name}: level must be in 0..{n - 1}, got {level}")

    def param(self):
        return self.corruption.severity_params[self.level]

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        if self.name == "motion_blur":
            radius, sigma = self.param()
            arr = _motion_blur(img, radius, sigma)
        else:
            arr = np.asarray(self.corruption.apply(img, severity=self.level))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape != (img.height, img.width, 3):
            raise RuntimeError(f"{self.name} level {self.level} returned shape {arr.shape} for a "
                               f"{img.width}x{img.height} image")
        return Image.fromarray(arr.astype(np.uint8))

    def __repr__(self):
        return f"MedMNISTCAugmentation({self.name}, level={self.level}, param={self.param()})"


class SpeckleStd:
    """Same formula as SpeckleNoise in pretrain_dataloader.py at sharing r = 0."""

    def __init__(self, std: float):
        self.std = float(std)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr + arr * np.random.randn(*arr.shape) * self.std, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))

    def __repr__(self):
        return f"SpeckleStd({self.std})"


class ContrastFactor:
    """Same formula as ContrastDecrease in pretrain_dataloader.py at sharing r = 0."""

    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        mean = arr.mean()
        arr = np.clip(mean + self.factor * (arr - mean), 0.0, 255.0)
        return Image.fromarray(arr.astype(np.uint8))

    def __repr__(self):
        return f"ContrastFactor({self.factor})"


class RandomApplyP:
    """Apply `fn` with probability p (no torch dependency)."""

    def __init__(self, fn, p: float):
        self.fn, self.p = fn, float(p)

    def __call__(self, img):
        return self.fn(img) if random.random() < self.p else img

    def __repr__(self):
        return f"RandomApplyP(p={self.p}, {self.fn!r})"


class GaussianRange:
    """Gaussian blur with sigma drawn uniformly from [lo, hi] (same filter as GaussianBlur in the dataloader)."""

    def __init__(self, lo: float, hi: float):
        self.lo, self.hi = float(lo), float(hi)

    def __call__(self, img: Image.Image) -> Image.Image:
        from PIL import ImageFilter
        return img.convert("RGB").filter(ImageFilter.GaussianBlur(random.uniform(self.lo, self.hi)))

    def __repr__(self):
        return f"GaussianRange({self.lo}, {self.hi})"


class JpegQuality:
    """JPEG compression at a fixed quality (same procedure as medmnistc, any quality)."""

    def __init__(self, quality: int):
        self.quality = int(quality)

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def __repr__(self):
        return f"JpegQuality({self.quality})"


class PixelateFactor:
    """Pixelation by a resize factor (same procedure as medmnistc, any factor)."""

    def __init__(self, factor: float):
        self.factor = float(factor)

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        small = img.resize((max(1, int(w * self.factor)), max(1, int(h * self.factor))), Image.BOX)
        return small.resize((w, h), Image.BOX)

    def __repr__(self):
        return f"PixelateFactor({self.factor})"


class MotionBlurCustom:
    """ImageMagick motion blur with explicit radius, sigma and a limited angle range (degrees, +/- angle)."""

    def __init__(self, radius: float, sigma: float, angle: float = 15.0):
        self.radius, self.sigma, self.angle = float(radius), float(sigma), float(angle)

    def __call__(self, img: Image.Image) -> Image.Image:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        m = MotionImage(blob=buf.getvalue())
        m.motion_blur(radius=self.radius, sigma=self.sigma, angle=np.random.uniform(-self.angle, self.angle))
        out = cv2.imdecode(np.frombuffer(m.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)
        buf.close()
        out = np.stack([out, out, out], axis=-1) if out.ndim == 2 else out[..., [2, 1, 0]]
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def __repr__(self):
        return f"MotionBlurCustom(radius={self.radius}, sigma={self.sigma}, angle=+/-{self.angle})"


# Order in which the new augmentations are appended after speckle and contrast.
EXTRA_ORDER = ["blur_custom", "motion_blur", "pixelate", "jpeg_compression"]


def build_extra_augmentations(cfg, dataset: str = "octmnist"):
    """Return the list of new augmentations that are switched on in `cfg`.

    Each has its own block in the augmentation yaml; a block with prob 0 (or missing) is skipped:
        blur_custom:      {prob, sigma_lo, sigma_hi}              Gaussian blur, sigma ~ U[lo, hi]
        motion_blur:      {prob, level} or {prob, radius, sigma, angle}   MedMNIST-C level index 0..4, or explicit values
        pixelate:         {prob, level} or {prob, factor}
        jpeg_compression: {prob, level} or {prob, quality}
    """
    out = []
    for name in EXTRA_ORDER:
        c = cfg.get(name, None)
        if c is None or not c.get("prob", 0) > 0:
            continue
        if name == "blur_custom":
            fn = GaussianRange(c.get("sigma_lo", 0.1), c["sigma_hi"])
        elif name == "motion_blur" and c.get("radius") is not None:
            fn = MotionBlurCustom(c["radius"], c["sigma"], c.get("angle", 15.0))
        elif name == "pixelate" and c.get("factor") is not None:
            fn = PixelateFactor(c["factor"])
        elif name == "jpeg_compression" and c.get("quality") is not None:
            fn = JpegQuality(c["quality"])
        else:
            fn = MedMNISTCAugmentation(name, c.get("level", 0), dataset)
        out.append(RandomApplyP(fn, c["prob"]))
    return out
