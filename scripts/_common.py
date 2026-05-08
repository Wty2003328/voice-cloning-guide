"""Shared setup for all scripts: locates the GPT-SoVITS repo, sets up paths, env vars."""
import os
import sys
from pathlib import Path

DEFAULT_GS_DIR = Path(os.environ.get("GS_DIR", "./GPT-SoVITS")).resolve()


def setup(gs_dir: Path = DEFAULT_GS_DIR, version: str = "v2") -> Path:
    """Add GPT-SoVITS to sys.path and set required env vars. Returns the resolved gs_dir."""
    gs_dir = Path(gs_dir).resolve()
    if not gs_dir.exists():
        raise FileNotFoundError(
            f"GPT-SoVITS not found at {gs_dir}. "
            f"Clone it: git clone https://github.com/RVC-Boss/GPT-SoVITS, "
            f"then point GS_DIR env var or --gs-dir at it."
        )
    os.chdir(str(gs_dir))
    sys.path.insert(0, str(gs_dir))
    sys.path.insert(0, str(gs_dir / "GPT_SoVITS"))
    os.environ.setdefault("version", version)
    os.environ.setdefault("hz", "25hz")
    return gs_dir


def pretrained_paths(gs_dir: Path):
    """Standard paths to the v2 pretrained models."""
    base = Path(gs_dir) / "GPT_SoVITS" / "pretrained_models"
    return {
        "s2g": str(base / "gsv-v2final-pretrained" / "s2G2333k.pth"),
        "s2d": str(base / "gsv-v2final-pretrained" / "s2D2333k.pth"),
        "s1": str(base / "gsv-v2final-pretrained"
                       / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
        "cnhubert": str(base / "chinese-hubert-base"),
        "bert": str(base / "chinese-roberta-wwm-ext-large"),
        "s2_config": str(Path(gs_dir) / "GPT_SoVITS" / "configs" / "s2.json"),
    }
