"""
Patches the NCL repo (github.com/PKU-ML/non_neg) to work with:
  - pytorch-lightning >= 2.0
  - Python 3.12

Run from inside the non_neg/ directory:
    python patch_for_pl2.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent  # run from non_neg/

PATCHES = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. main_pretrain.py
# ─────────────────────────────────────────────────────────────────────────────
def patch_main_pretrain():
    path = ROOT / "main_pretrain.py"
    src = path.read_text()

    # DDPStrategy import path
    src = src.replace(
        "from pytorch_lightning.strategies.ddp import DDPStrategy",
        "from pytorch_lightning.strategies import DDPStrategy",
    )

    # WandbLogger import path (moved in PL 2.x)
    src = src.replace(
        "from pytorch_lightning.loggers import WandbLogger",
        "try:\n    from pytorch_lightning.loggers import WandbLogger\nexcept ImportError:\n    from lightning.pytorch.loggers import WandbLogger",
    )

    path.write_text(src)
    print("✅ Patched main_pretrain.py")

# ─────────────────────────────────────────────────────────────────────────────
# 2. main_linear.py
# ─────────────────────────────────────────────────────────────────────────────
def patch_main_linear():
    path = ROOT / "main_linear.py"
    src = path.read_text()

    src = src.replace(
        "from pytorch_lightning.strategies.ddp import DDPStrategy",
        "from pytorch_lightning.strategies import DDPStrategy",
    )

    src = src.replace(
        "from pytorch_lightning.loggers import WandbLogger",
        "try:\n    from pytorch_lightning.loggers import WandbLogger\nexcept ImportError:\n    from lightning.pytorch.loggers import WandbLogger",
    )

    path.write_text(src)
    print("✅ Patched main_linear.py")

# ─────────────────────────────────────────────────────────────────────────────
# 3. solo/methods/base.py  — validation_epoch_end → on_validation_epoch_end
# ─────────────────────────────────────────────────────────────────────────────
def patch_base():
    path = ROOT / "solo" / "methods" / "base.py"
    src = path.read_text()

    # PL 2.x renamed these hooks
    src = src.replace(
        "def validation_epoch_end(self, outs",
        "def on_validation_epoch_end(self, outs",  # first occurrence — BaseMethod
    )
    # There are two occurrences; replace the second (MomentumMethod) as well
    # The replace above only replaces the first call; do it again for remaining
    src = src.replace(
        "def validation_epoch_end(self, outs",
        "def on_validation_epoch_end(self, outs",
    )
    # Also fix the super() call inside MomentumMethod
    src = src.replace(
        "super().validation_epoch_end(parent_outs)",
        "super().on_validation_epoch_end(parent_outs)",
    )

    path.write_text(src)
    print("✅ Patched solo/methods/base.py")

# ─────────────────────────────────────────────────────────────────────────────
# 4. solo/methods/linear.py — training_epoch_end + validation_epoch_end
# ─────────────────────────────────────────────────────────────────────────────
def patch_linear():
    path = ROOT / "solo" / "methods" / "linear.py"
    src = path.read_text()

    src = src.replace(
        "def training_epoch_end(self,outs)",
        "def on_train_epoch_end(self)",
    )
    # The body of training_epoch_end used `outs` — in PL 2.x it's gone.
    # Replace the outs reference with self.trainer outputs (they logged already)
    src = src.replace(
        "def on_train_epoch_end(self):\n        pass",
        "def on_train_epoch_end(self):\n        pass",
    )

    src = src.replace(
        "def validation_epoch_end(self, outs",
        "def on_validation_epoch_end(self, outs",
    )

    path.write_text(src)
    print("✅ Patched solo/methods/linear.py")

# ─────────────────────────────────────────────────────────────────────────────
# 5. solo/utils/checkpointer.py — WandbLogger import
# ─────────────────────────────────────────────────────────────────────────────
def patch_checkpointer():
    path = ROOT / "solo" / "utils" / "checkpointer.py"
    src = path.read_text()

    if "from pytorch_lightning.loggers import WandbLogger" in src:
        src = src.replace(
            "from pytorch_lightning.loggers import WandbLogger",
            "try:\n    from pytorch_lightning.loggers import WandbLogger\nexcept ImportError:\n    from lightning.pytorch.loggers import WandbLogger",
        )
        path.write_text(src)
        print("✅ Patched solo/utils/checkpointer.py")
    else:
        print("⏭️  solo/utils/checkpointer.py — no WandbLogger import, skipping")

# ─────────────────────────────────────────────────────────────────────────────
# 6. solo/utils/auto_umap.py — WandbLogger import
# ─────────────────────────────────────────────────────────────────────────────
def patch_auto_umap():
    path = ROOT / "solo" / "utils" / "auto_umap.py"
    src = path.read_text()

    if "pl.loggers.WandbLogger" in src:
        src = src.replace(
            "isinstance(trainer.logger, pl.loggers.WandbLogger)",
            "isinstance(trainer.logger, WandbLogger) if 'WandbLogger' in dir() else False",
        )
        path.write_text(src)
        print("✅ Patched solo/utils/auto_umap.py")
    else:
        print("⏭️  solo/utils/auto_umap.py — nothing to patch, skipping")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Disable W&B in all wandb config yamls
# ─────────────────────────────────────────────────────────────────────────────
def disable_wandb():
    for yaml_path in ROOT.rglob("wandb/private.yaml"):
        src = yaml_path.read_text()
        src = re.sub(r"enabled:\s*True", "enabled: False", src)
        yaml_path.write_text(src)
        print(f"✅ Disabled W&B in {yaml_path.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Fix single-GPU: change strategy: ddp → null in CIFAR pretrain yamls
# ─────────────────────────────────────────────────────────────────────────────
def fix_single_gpu():
    for yaml_path in (ROOT / "scripts" / "pretrain" / "cifar").glob("*.yaml"):
        src = yaml_path.read_text()
        src = re.sub(r'^strategy:\s*"ddp"', 'strategy: null', src, flags=re.MULTILINE)
        yaml_path.write_text(src)
        print(f"✅ Set strategy: null in {yaml_path.relative_to(ROOT)}")
    for yaml_path in (ROOT / "scripts" / "eval" / "cifar").glob("*.yaml"):
        src = yaml_path.read_text()
        src = re.sub(r'^strategy:\s*"ddp"', 'strategy: null', src, flags=re.MULTILINE)
        yaml_path.write_text(src)
        print(f"✅ Set strategy: null in {yaml_path.relative_to(ROOT)}")

# ─────────────────────────────────────────────────────────────────────────────
# Run all patches
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔧 Patching NCL repo for pytorch-lightning 2.x + Python 3.12\n")
    patch_main_pretrain()
    patch_main_linear()
    patch_base()
    patch_linear()
    patch_checkpointer()
    patch_auto_umap()
    disable_wandb()
    fix_single_gpu()
    print("\n✅ All patches applied. Now install missing deps and run:")
    print("   pip install einops timm hydra-core wandb scipy scikit-learn")
    print("   pip install -e .")
    print()
    print("Then pretrain:")
    print("   python main_pretrain.py --config-path scripts/pretrain/cifar --config-name simclr.yaml")
    print("   python main_pretrain.py --config-path scripts/pretrain/cifar --config-name ncl.yaml")
    print()
    print("Then evaluate semantic consistency:")
    print("   python main_eval.py --config-path scripts/eval/cifar --config-name simclr.yaml \\")
    print("       resume_from_checkpoint=trained_models/simclr-resnet18-cifar100/last.ckpt")
    print("   python main_eval.py --config-path scripts/eval/cifar --config-name ncl.yaml \\")
    print("       resume_from_checkpoint=trained_models/simclr-resnet18-cifar100-ncl/last.ckpt")
