"""Shared setup for all scripts: locates the GPT-SoVITS repo, sets up paths, env vars."""
import os
import sys
from pathlib import Path

DEFAULT_GS_DIR = Path(os.environ.get("GS_DIR", "./GPT-SoVITS")).resolve()


def setup(gs_dir: Path = DEFAULT_GS_DIR, version: str = "v4") -> Path:
    """Add GPT-SoVITS to sys.path and set required env vars. Returns the resolved gs_dir.

    Default version is v4 — the canonical fine-tune path documented in
    docs/models/gpt-sovits-v4.md. v2 is no longer supported by these
    scripts (the v2-specific entries were removed when the guide
    consolidated on v4 in 2026-05).
    """
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


def pretrained_paths(gs_dir: Path, version: str = "v4"):
    """Pretrained model paths for GPT-SoVITS v4 — the canonical version
    documented in this guide. (v2 support removed 2026-05.)
    """
    if version != "v4":
        raise ValueError(
            f"Unsupported version: {version!r}. This guide consolidated on "
            f"v4 in 2026-05; v2 scripts were removed. Use version='v4'."
        )
    base = Path(gs_dir) / "GPT_SoVITS" / "pretrained_models"
    return {
        "cnhubert": str(base / "chinese-hubert-base"),
        "bert": str(base / "chinese-roberta-wwm-ext-large"),
        "s2_config": str(Path(gs_dir) / "GPT_SoVITS" / "configs" / "s2.json"),
        "s2g": str(base / "gsv-v4-pretrained" / "s2Gv4.pth"),
        "vocoder": str(base / "gsv-v4-pretrained" / "vocoder.pth"),
        "s1": str(base / "s1v3.ckpt"),
    }
