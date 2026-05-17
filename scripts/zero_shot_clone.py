"""Zero-shot voice cloning with Qwen3-TTS-12Hz-1.7B-Base.

Minimal end-to-end inference: load model → register reference clip →
synthesize one or many target texts → save WAV. Mirrors the
production wrapper used in waifu-companion's qwen3_tts_sidecar.

Hybrid speaker conditioning: same-language target uses paired prompt
features; cross-lingual target uses x_vector_only mode. See
docs/10-zero-shot-cloning.md for the why.

Usage:
    python zero_shot_clone.py \\
        --model-dir ./qwen3-tts-1.7b-base \\
        --reference my_voice.wav \\
        --reference-text "Hello, this is my reference recording." \\
        --reference-lang en \\
        --text "I can speak any text now." \\
        --language en \\
        --out cloned.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

# ── Language map: API expects names, not BCP-47 codes ─────────────────
LANG_HINT = {
    "ja": "Japanese", "en": "English", "zh": "Chinese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian",
    "pt": "Portuguese", "es": "Spanish", "it": "Italian",
}

# ── Quality preset → native sampling params ────────────────────────────
QUALITY_PRESETS = {
    "fast":     {"temperature": 0.6, "top_p": 0.70, "max_new_tokens": 200},
    "balanced": {"temperature": 0.4, "top_p": 0.85, "max_new_tokens": 240},
    "high":     {"temperature": 0.3, "top_p": 0.90, "max_new_tokens": 320},
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True,
                   help="Local path to Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--reference", required=True,
                   help="Reference audio WAV (3-32s of the voice to clone)")
    p.add_argument("--reference-text", required=True,
                   help="Exact transcript of the reference audio")
    p.add_argument("--reference-lang", default="ja",
                   choices=sorted(LANG_HINT.keys()),
                   help="Language of the reference audio (BCP-47)")
    p.add_argument("--text", required=True,
                   help="Target text to synthesize")
    p.add_argument("--language", required=True,
                   choices=sorted(LANG_HINT.keys()),
                   help="Target language (BCP-47)")
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--quality", default="balanced",
                   choices=["fast", "balanced", "high"])
    p.add_argument("--attn", default="sdpa",
                   choices=["auto", "sdpa", "flash_attention_2", "manual"])
    p.add_argument("--dtype", default="bf16",
                   choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: --model-dir not found: {model_dir}", file=sys.stderr)
        return 2
    ref = Path(args.reference)
    if not ref.exists():
        print(f"ERROR: --reference not found: {ref}", file=sys.stderr)
        return 2

    # ── Load model ────────────────────────────────────────────────────
    print(f"[load] {model_dir} (dtype={args.dtype}, attn={args.attn})",
          flush=True)
    from qwen_tts import Qwen3TTSModel
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}
    kwargs = dict(device_map="cuda:0", dtype=dtype_map[args.dtype])
    if args.attn == "auto":
        try:
            import flash_attn  # noqa
            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    elif args.attn in ("sdpa", "flash_attention_2"):
        kwargs["attn_implementation"] = args.attn
    model = Qwen3TTSModel.from_pretrained(str(model_dir), **kwargs)

    # ── Build hybrid prompt cache ─────────────────────────────────────
    # x_vector_only=False  → better same-language clone fidelity
    # x_vector_only=True   → better cross-lingual (less reference-lang bleed)
    print(f"[prompt] building hybrid prompts for {ref.name}", flush=True)
    is_cross_lingual = (args.language != args.reference_lang)
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(ref),
        ref_text=args.reference_text,
        x_vector_only_mode=is_cross_lingual,
    )

    # ── Synthesize ────────────────────────────────────────────────────
    preset = QUALITY_PRESETS[args.quality]
    print(f"[synth] '{args.text[:60]}...' -> {args.language} "
          f"({'cross-lingual' if is_cross_lingual else 'same-lang'}, "
          f"quality={args.quality})", flush=True)
    wavs, sr = model.generate_voice_clone(
        text=args.text,
        language=LANG_HINT[args.language],
        voice_clone_prompt=prompt,
        **preset,
    )

    # ── Save ──────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), wavs[0], sr)
    print(f"[done] wrote {out} ({len(wavs[0])/sr:.2f}s @ {sr} Hz)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
