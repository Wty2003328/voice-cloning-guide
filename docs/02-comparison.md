# 02 — How GPT-SoVITS Compares to Other Voice Cloning Models

A pragmatic comparison of the dominant open-source voice cloning systems as of 2026.

## At a glance

| Model | Type | Min audio | Quality | Speed | Languages | Windows-friendly |
|---|---|---|---|---|---|---|
| **GPT-SoVITS v2** | Fine-tune TTS | 1 min | High | Fast (RT) | zh / ja / en / ko / yue | ✅ Yes |
| **GPT-SoVITS v4** | Fine-tune TTS | 1 min | Higher (48k) | Fast | same | ✅ Yes |
| **RVC v2** | Voice conversion | 10 min | High | Real-time | Any (timbre only) | ✅ Yes |
| **CosyVoice 2/3** | Fine-tune TTS | 5 min | High | Medium | zh / en | ⚠️ Linux preferred (deepspeed) |
| **XTTS v2** | Zero-shot TTS | 10s ref | Medium-high | Slow | 17 languages | ✅ Yes |
| **Bark** | Zero-shot TTS | None | Low-Medium | Very slow | Multilingual | ✅ Yes |
| **Fish Speech** | Fine-tune TTS | 1 min | High | Fast | zh / en / ja | ✅ Yes |
| **VoiceCraft** | Zero-shot TTS / edit | 3s ref | Medium | Medium | English-focused | ⚠️ Some bugs |

"Fine-tune TTS" means you train on a target speaker before inference. "Zero-shot TTS" means you supply a short reference at inference time only. "Voice conversion" means you supply a separate TTS as input and the model warps the timbre to match the target.

## When to use each

### GPT-SoVITS — best default for character voice cloning

Use it when:
- You want **anime / game / streamer** style voices — the kind of expressive, stylized speech that needs the model to learn prosody patterns, not just timbre.
- You have **1-30 minutes** of training audio available.
- You need **multi-lingual output**: train on Japanese, generate in English / Chinese / Korean.
- You're on **Windows** with a single GPU.

Skip it when:
- You only have a 10-second reference and can't get more — use XTTS or VoiceCraft.
- You need real-time voice conversion of live input — use RVC.
- Your target language isn't in zh/ja/en/ko/yue (the v2 phonemizer set).

### RVC v2 — best for live voice conversion (singing, streaming)

RVC is fundamentally different: it doesn't synthesize speech from text. Instead, it takes existing audio (e.g., your microphone, a song you sang) and re-renders it in a target voice. This makes it the only good choice for:
- Singing voice cloning (RVC retains pitch perfectly).
- Real-time conversion during live streams.
- Use cases where you already have a TTS you like and just want to swap the timbre.

Drawback: RVC only swaps timbre. The prosody (rhythm, emphasis, pacing) comes from the input audio. If you want the *speaker's mannerisms* — pauses, sigh-y exhales, rising sentence endings — RVC can't give you those. GPT-SoVITS can.

A common production pipeline used to be **TTS + RVC** (any TTS for content, RVC for timbre conversion). With fine-tuned GPT-SoVITS, this is mostly obsolete: a single fine-tuned GPT-SoVITS captures both prosody and timbre, with no quality loss from chaining models, and is faster.

### CosyVoice 2/3 — best zero-shot quality

If you only have a short reference and want production quality without fine-tuning, CosyVoice 2/3 is hard to beat. The flow-matching architecture produces remarkably clean output and handles emotion conditioning well.

Drawback: the fine-tuning code path requires DeepSpeed, which is painful on Windows. If you're committing to a fine-tune workflow, GPT-SoVITS is more accessible. If you want zero-shot, CosyVoice usually wins on naturalness.

### XTTS v2 — best for breadth of language support

17 languages out of the box, including European languages GPT-SoVITS doesn't handle. Zero-shot only — fine-tuning isn't really supported.

Drawback: quality ceiling is noticeably lower than GPT-SoVITS or CosyVoice for the languages all three support, and prosody is somewhat flat.

### Bark — historical interest mostly

Bark introduced zero-shot voice cloning to the open-source world but has been superseded on every dimension: slower, lower quality, less controllable. Skip unless you need the specific Bark "music + voice" capability.

### Fish Speech — close GPT-SoVITS competitor

Architectural cousin of GPT-SoVITS (also a two-stage GPT + decoder design). Comparable quality. Slightly different language coverage (no Korean). Fine to use either one — pick by which has better tooling for your use case. As of 2026 GPT-SoVITS has more community resources and tutorials (this repo being one).

### VoiceCraft — speech editing niche

VoiceCraft's strength is that it can edit existing speech — splice in new words while matching the surrounding voice. Useful for correcting recordings. As a general TTS it's solid but unremarkable.

## Why this guide picks GPT-SoVITS for fine-tuning

Three reasons specifically:

1. **Single GPU, Windows-friendly**. Most modern TTS models assume Linux + multi-GPU + DDP. GPT-SoVITS works fine on a single consumer card with a few small bypasses (this repo's job is to document those).

2. **Multi-lingual output from monolingual training**. Train on 15 minutes of Japanese, get usable English and Chinese output. CosyVoice can do this too; XTTS cannot.

3. **The model architecture is small enough to teach**. Two clean components (GPT + SoVITS) with explicit data flow. CosyVoice's DiT + flow-matching is mathematically cleaner but harder to debug; XTTS hides everything behind a black-box CLI.

## What about TTS+RVC (the historical pipeline)?

Older voice cloning tutorials chain a generic TTS (often XTTS or Bark) into RVC for timbre conversion. This was a workaround for the lack of good single-stage fine-tunable TTS. Now that GPT-SoVITS exists, the chained approach is mostly worse:

- **More compute.** Two model loads, two inference passes per request.
- **Quality degradation at the boundary.** RVC is good but not transparent — you lose some fidelity at the conversion step.
- **Worse prosody.** RVC inherits whatever the upstream TTS produces. Fine-tuned GPT-SoVITS produces prosody specific to your target speaker.
- **Hard to coordinate languages.** If RVC is trained on Japanese audio but the input TTS is English, the conversion can produce artifacts.

Use TTS+RVC only when you need **real-time** conversion (live mic input) and can't pre-render the speech.

## Further reading

- The CosyVoice paper for a great writeup of flow-matching TTS: https://arxiv.org/abs/2407.05407
- RVC's GitHub for the voice-conversion approach: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- A broader survey of voice cloning: https://arxiv.org/abs/2505.00579
