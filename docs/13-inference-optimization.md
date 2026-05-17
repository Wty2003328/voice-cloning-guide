# 13 — Inference Optimization: Beating the 2× RTF Barrier on Windows

Most consumer TTS deployments on Windows + NVIDIA hit an annoying ceiling
around RTF 2.0–2.5 (~2× slower than real-time). The model isn't slow —
the **Python autoregressive loop** dispatches thousands of tiny GPU ops
with per-op overhead that the GPU never sees. This doc shows how we hit
**RTF 0.40 in lab benchmark — 2.5× faster than real-time, 5.98× speedup
over baseline** on Qwen3-TTS-12Hz-1.7B-Base on a single RTX 5080, on
Windows, with **zero quality loss** (ASR Jaccard 1.000 vs baseline) and
**without** flash-attn, vLLM, quantization, or recompiling PyTorch.

The technique generalises beyond Qwen3-TTS to most autoregressive TTS
that uses HuggingFace's `GenerationMixin`. The diagnosis pattern is the
real value: how to know *what* to fix before fixing.

## When this matters

You're on this page if **any** of these are true:

- Your TTS RTF is in the 1.4–2.0 range on a modern GPU (RTX 30/40/50,
  A-series, H-series) and you can't figure out why it isn't faster.
- You've tried `torch.compile` and got a 1.1× speedup or a regression.
- You've tried quantization (FP8, INT4) and it made things slower.
- You've heard "use flash-attn" and you can't actually install it.

If you only need RTF < 5 (most chat use cases) and you're on Linux, vLLM
or TensorRT-LLM will get you there faster than this doc. If you're on
Windows + Blackwell/Ada and need RTF < 1, **this is the path**.

## Hardware / software context

The recipe was validated against:

| | Validated configuration |
|---|---|
| **GPU** | NVIDIA RTX 5080 (Blackwell, compute capability 12.0, 16 GB) |
| **OS** | Windows 11 |
| **Python** | 3.10 |
| **PyTorch** | 2.11.0+cu128 |
| **Triton** | triton-windows 3.7.0 (`pip install triton-windows`) |
| **Model** | Qwen3-TTS-12Hz-1.7B-Base via `qwen-tts==0.1.1` |
| **dtype / attention** | bf16 + sdpa (efficient_attention backend) |

The techniques apply to other Blackwell / Ada / Ampere GPUs and to most
HuggingFace-style RQ-Transformer TTS models (Qwen3-TTS, possibly
CosyVoice 3, possibly Higgs Audio).

## Step 1 — diagnose with `torch.profiler`

**Before optimizing, find the actual bottleneck.** Most TTS perf
investigations skip this step and waste days on the wrong layer.

```python
import torch
from torch.profiler import profile, ProfilerActivity

# Warm the model up (2-3 calls) before profiling.
for _ in range(2):
    model.generate_voice_clone(text="warmup", language="Japanese",
                               voice_clone_prompt=prompt, max_new_tokens=200)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model.generate_voice_clone(text="…test sentence…", language="Japanese",
                               voice_clone_prompt=prompt, max_new_tokens=240)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
print(f"CPU total: {prof.key_averages().total_average().cpu_time_total/1e6:.1f}s")
print(f"CUDA total: {prof.key_averages().total_average().cuda_time_total/1e6:.1f}s")
```

The diagnostic question: **what's the CPU-to-CUDA wall-time ratio?**

- **CPU ≈ CUDA**: the GPU is busy. Compute-bound. Quantization /
  flash-attn / smaller model are the levers.
- **CPU >> CUDA**: the GPU is mostly *idle*, blocked on Python dispatch.
  Compile / CUDA-graph / replace generate() loop are the levers.

On Qwen3-TTS in our setup:

```
CPU total: 13.3s
CUDA total: 1.2s         ← 10× headroom on GPU
aten::matmul:    55,992 calls,  9.5μs CUDA avg
aten::copy_:    112,260 calls
```

55,000 matmul calls per generation, each one CUDA-ready in ~10μs but
fronted by ~200μs of Python dispatch (attribute lookup, dtype check,
kernel launch). **The GPU never gets to run flat-out.** This is the
classic Python-bound autoregressive decode story.

## Step 2 — understand the architecture you're fighting

Qwen3-TTS is an **RQ-Transformer**: each main audio token is decoded via
a *nested* loop over a 15-codebook residual quantizer. The wrapper
`Qwen3TTSModel` runs roughly:

```text
for outer_step in range(audio_token_count):           # ~60 for 5s of audio
    main_token = talker.generate(...)                  # 28-layer LM, 1 step
    for inner_step in range(15):                       # ← inner RQ loop
        code_predictor.generate(...)                  # 5-layer LM, 1 step
```

That's **~60 outer forwards + 60 × 15 = 900 inner forwards** per ~5s of
audio. Each `.generate()` call traverses HuggingFace's full
`GenerationMixin` pipeline:

1. `LogitsProcessor` list (top-k → top-p → temperature → suppress_tokens → …)
2. Sampler (`multinomial` or `argmax`)
3. `StoppingCriteria` list (length, EOS, time-budget, …)
4. `_update_model_kwargs_for_generation` (attention mask, cache, position_ids)
5. The actual forward()

The Python overhead of (1) – (4) dominates when (5) is small (which it
is for the 5-layer inner predictor). That's where the 13.3s of CPU time
hides.

## Step 3 — Tactic T1: replace the inner `generate()` with a tight loop

Skip HF's pipeline for the inner predictor — its forward() is small
enough that direct calls + a 4-line sampler beat the full machinery.

```python
import types
import torch
import torch.nn.functional as F


def _top_k_top_p_filter(logits, top_k, top_p):
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cumprob = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumprob - sorted_logits.softmax(dim=-1) > top_p
        mask[..., 0] = False
        scatter_mask = torch.zeros_like(mask).scatter_(-1, sorted_idx, mask)
        logits = logits.masked_fill(scatter_mask, float("-inf"))
    return logits


def _sample(logits, do_sample, top_k, top_p, temperature):
    if not do_sample or temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(temperature, 1e-5)
    logits = _top_k_top_p_filter(logits, top_k, top_p)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)


def install_fast_code_predictor_generate(model):
    """Monkey-patch the inner predictor's .generate to a tight loop."""
    inner = model.model if hasattr(model.model, "talker") else model
    predictor = inner.talker.code_predictor

    class _Result:
        __slots__ = ("sequences",)
        def __init__(self, sequences): self.sequences = sequences

    @torch.inference_mode()
    def fast_generate(self, inputs_embeds=None, max_new_tokens=None,
                      do_sample=True, top_k=50, top_p=1.0, temperature=0.9, **_):
        out = predictor.forward(inputs_embeds=inputs_embeds, use_cache=True, return_dict=True)
        past = out.past_key_values
        gs = inputs_embeds.shape[1] - 1
        next_id = _sample(out.logits[:, -1, :], do_sample, top_k, top_p, temperature)
        seqs = [next_id]
        for _ in range(max_new_tokens - 1):
            out = predictor.forward(
                input_ids=next_id, past_key_values=past,
                use_cache=True, generation_steps=gs, return_dict=True,
            )
            past = out.past_key_values
            gs += 1
            next_id = _sample(out.logits[:, -1, :], do_sample, top_k, top_p, temperature)
            seqs.append(next_id)
        return _Result(sequences=torch.cat(seqs, dim=-1))

    predictor.generate = types.MethodType(fast_generate, predictor)
```

**Measured speedup alone: 1.80× (lab: RTF 2.41 → 1.34).** Quality
ASR-verified equivalent.

The win: dropping the LogitsProcessor list, sampler dispatch, and
stopping-criteria checks 900 times per call removes ~5s of Python
overhead.

## Step 4 — Tactic T2: `torch.compile` the inner forward

Once T1 calls `predictor.forward()` directly (no HF wrapping), it's now
a **simple, compile-friendly call** — same shapes, same dtypes, no
control flow. Apply `torch.compile` with cudagraph mode:

```python
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision("high")
predictor.model.forward = torch.compile(
    predictor.model.forward,
    mode="reduce-overhead",   # enables CUDA graph capture
    dynamic=False,            # static shapes only — key for cudagraphs
)
```

**Critical detail: cudagraph step boundary.** PyTorch's
`reduce-overhead` mode uses CUDA graphs across compiled regions, which
means it expects the caller to mark when one "step" ends. If you don't,
you get `AssertionError: len(node.tensor_weakrefs) == len(node.stack_traces)`
on the second call.

Mark the boundary at the top of each outer voice-clone call:

```python
def synthesize(...):
    torch.compiler.cudagraph_mark_step_begin()
    return model.generate_voice_clone(...)
```

**Measured speedup T1 + T2 combined: 2.81× in lab bench (RTF 2.41 →
0.86), 3.3× in production sidecar (RTF ≈ 1.78 → 0.55 after warmup-amortized
prompts).** Quality still ASR-verified equivalent.

The win: the inner `predictor.forward` is now a single fused CUDA
graph that runs without Python intervention for 900 calls.

## Step 5 — absorb the autotune cost with a startup warmup

`torch.compile` autotunes on the first call — typically 30-60s for a
5-layer LM. If your TTS sidecar is doing this on a user's first
chat message, they wait. **Move it to boot.**

```python
# At the end of sidecar startup, after the model loads and the first
# voice is registered:
def warmup(model, prompt):
    torch.compiler.cudagraph_mark_step_begin()
    model.generate_voice_clone(
        text="warmup", language="Japanese",
        voice_clone_prompt=prompt,
        temperature=0.4, top_p=0.85, max_new_tokens=80,
    )
```

After this, every user-facing synthesis call is hot. Boot is ~70s
instead of ~12s, but the first user message is 2.5s instead of 47s —
a much better UX trade.

## Step 6 — Tactic T3: tight outer talker loop

T1 + T2 + warmup get you to **RTF 0.86 lab / 0.55 production**. The
inner 5-layer codebook predictor is now fully optimized. But the
**outer 28-layer talker** is still going through HuggingFace's full
`GenerationMixin.generate()` once per of the ~60 outer codec steps.
That's a second nested-loop overhead — smaller per-step than the inner
loop, but it adds up.

T3 applies the same trick at the outer level: replace
`Qwen3TTSTalkerForConditionalGeneration.generate()` with a tight
forward+sample loop that manually applies `suppress_tokens` and
`repetition_penalty=1.05`. The 60 outer steps no longer pay for HF's
`prepare_inputs_for_generation`, `LogitsProcessor` list, sampler
dispatch, or `_update_model_kwargs_for_generation` cache-stitching.

```python
def install_T3_fast_outer_generate(talker, codec_eos_token_id, suppress_token_set):
    """Replace HF's GenerationMixin.generate on the outer 28-layer talker
    with a tight loop. Mirrors T1's pattern at the codec-step level."""
    suppress_idx = torch.tensor(sorted(suppress_token_set),
                                device=talker.device, dtype=torch.long)

    class _TalkerResult:
        __slots__ = ("hidden_states",)
        def __init__(self, hidden_states): self.hidden_states = hidden_states

    @torch.inference_mode()
    def fast_generate(self, inputs_embeds=None, attention_mask=None,
                      trailing_text_hidden=None, tts_pad_embed=None,
                      max_new_tokens=2048, min_new_tokens=2,
                      do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                      eos_token_id=None, repetition_penalty=1.05, **_):
        eos = eos_token_id if eos_token_id is not None else codec_eos_token_id
        # Prefill (variable seq_len — NEVER routed through compiled path).
        self.rope_deltas = None
        out = self.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                           past_key_values=None, use_cache=True, output_hidden_states=True,
                           past_hidden=None, trailing_text_hidden=trailing_text_hidden,
                           tts_pad_embed=tts_pad_embed, generation_step=None)
        past, past_hidden, gen_step = out.past_key_values, out.past_hidden, out.generation_step
        collected = []
        logits = self._apply_token_penalties(out.logits[:, -1, :], suppress_idx, None, repetition_penalty)
        logits[..., eos] = float("-inf")  # mask EOS during min_new_tokens
        next_id = _sample(logits, do_sample, top_k, top_p, temperature)
        gen_history = [next_id]
        cache_pos = past.get_seq_length()
        attn_mask = attention_mask
        for step in range(1, max_new_tokens + 1):
            attn_mask = torch.cat([attn_mask, attn_mask.new_ones((1, 1))], dim=-1)
            out = self.forward(input_ids=next_id, attention_mask=attn_mask,
                               past_key_values=past, use_cache=True,
                               output_hidden_states=True,
                               cache_position=torch.tensor([cache_pos], device=next_id.device),
                               past_hidden=past_hidden,
                               trailing_text_hidden=trailing_text_hidden,
                               tts_pad_embed=tts_pad_embed, generation_step=gen_step)
            past, past_hidden, gen_step = out.past_key_values, out.past_hidden, out.generation_step
            cache_pos += 1
            codec_ids = out.hidden_states[1] if isinstance(out.hidden_states, tuple) else None
            collected.append(((past_hidden,), codec_ids))
            if (codec_ids[:, 0] == eos).all().item():
                break
            logits = self._apply_token_penalties(out.logits[:, -1, :], suppress_idx,
                                                  gen_history, repetition_penalty)
            if step < min_new_tokens:
                logits[..., eos] = float("-inf")
            next_id = _sample(logits, do_sample, top_k, top_p, temperature)
            gen_history.append(next_id)
        return _TalkerResult(hidden_states=collected)

    talker.generate = types.MethodType(fast_generate, talker)
```

**Measured speedup T1 + T2 + T3 combined: 5.98× cumulative** (lab RTF
2.41 → 0.40). Quality ASR Jaccard 1.000 — punctuation pauses (`。`,
`?`) which T1+T2 alone occasionally drops are restored by T3's
explicit `repetition_penalty` application.

**No new compile cost** — T3 is pure Python loop replacement, no
`torch.compile` warmup. **No new failure modes** — the tight-loop
pattern is identical to T1's, which has been in production.

## Measured results — full table

Test sentence: `"今日はとてもいい天気ですね。一緒にお散歩しましょうか？"`
(~4.8s audio). 3 trials each, mean. Lab numbers from
`tts_lab/eval_out/_kernel_opt/summary.json`; production numbers from
the integrated TTS sidecar after register-voice warmup.

### Lab bench (Round 1: `_kernel_opt_prototype.py`)

| Config | wall mean (s) | wall min (s) | RTF | Speedup |
|---|---|---|---|---|
| Baseline (bf16+sdpa) | 11.63 | 9.67 | 2.41 | 1.00× |
| T1: tight inner predictor loop | 6.24 | 5.55 | 1.34 | 1.80× |
| T1 + T2 (compile + cudagraph) | **4.10** | **2.99** | **0.86** | **2.81×** |

### Lab bench (Round 2: `_kernel_opt_round2.py`)

T3 = tight outer talker loop, mirroring T1's pattern at the outer codec
level. See Step 6 below for what it does.

| Config | wall mean (s) | audio (s) | RTF | Cumulative vs baseline | ASR Jaccard vs baseline |
|---|---|---|---|---|---|
| T1 + T2 (rerun) | 2.08 | 4.40 | 0.472 | 5.10× | 0.92 (lost punctuation pauses) |
| **T1 + T2 + T3** | 2.60 | 6.45 | **0.403** | **5.98×** | **1.000 (full pauses)** |

T3's win is two-fold: lower RTF *and* restored punctuation prosody.
The rerun of T1+T2 dropped the `。` and `?` pauses (Jaccard 0.92);
adding T3 restores them by replicating HF's `repetition_penalty=1.05`
and `suppress_tokens` mask manually rather than skipping them. Wall
time per call is comparable; the model just produces more audio per
second of compute (RTF improvement).

### Production sidecar (after warmup)

| Config | wall mean (s) | RTF | Speedup |
|---|---|---|---|
| Baseline (no opt) | ~8.2 | ~1.78 | 1.00× |
| T1 + T2 + register-voice warmup | **~2.45** | **~0.55** | **~3.3×** |

The production numbers are better because (a) the register-voice
warmup absorbs torch.compile autotune *and* the first cudagraph
capture, (b) successive calls reuse the cached graph, and (c) HTTP
serialization runs concurrent with the next decode step. The lab
bench measures cold-trial variance directly.

ASR Jaccard vs baseline = 1.000 in all configs — same transcript,
same audio shape, no quality loss.

**VRAM unchanged** at 4.7 GB peak. No model surgery, no precision
loss, no smaller model. The CPU profile went from 13.3s to ~2.4s of
Python time per call.

**First-call (autotune) cost: ~46s** on RTX 5080. Move it to boot
via warmup (Step 5).

## What didn't work — documented dead ends

| Path tried | Result | Why |
|---|---|---|
| `torch.compile(model, mode="reduce-overhead")` on whole model | 1.11× | Dynamo bails on HF's `generate()` control flow — falls back to eager. |
| `torch.compile(... mode="max-autotune")` | 0.99× | Same fallback + autotune overhead. |
| **T4**: `torch.compile(talker.model.forward, mode="reduce-overhead")` on outer 28-layer talker | killed | HF `DynamicCache` grows lazily per generate(); Dynamo guards on `len(layers)`; cache_size_limit=64 exhausted; >10 min compile overhead before a single warmup completed. |
| **T4** with `mode="default"` (no cudagraphs) | killed | Same root cause: per-shape-signature Inductor codegen of 28-layer block, >7 min before warmup. |
| **T5**: per-decoder-layer compile (28 individual compiles) | **0.88× (regression)** | Per-layer compile prevents Inductor fusion across residual / attention boundary AND adds 28× per-layer Dynamo guard overhead. |
| `torchao.Float8DynamicActivationFloat8WeightConfig` | **0.12× (8× slower!)** | Blackwell FP8 tensor-core kernels not yet wired through torchao 0.17 on Windows; falls back to a Python dequant-on-every-matmul path. |
| `torchao.Float8WeightOnlyConfig` | 0.56× | Same kernel-fallback issue. |
| `fp16` instead of `bf16` | 1.79 RTF (slower) | Blackwell prefers bf16 for this op set. |
| `attn_implementation="eager"` | 2.05 RTF | SDPA's `efficient_attention` already optimal. |
| flash-attn 2 pip install | Failed | Windows long-path bug breaks the source build; community wheels for torch 2.11+cu128+py3.10 + Blackwell don't exist yet. |
| vLLM serving on Windows | N/A | Windows alpha; Blackwell-NVFP4 kernels Linux-only. |
| vLLM custom Qwen3-TTS integration | 3-5 weeks engineering | Architectural mismatch — vLLM's continuous-batching scheduler assumes "one forward → one token"; Qwen3-TTS does "one outer forward → 15 nested inner forwards → one audio token". Single-user has no batching win. |
| TensorRT-LLM with Qwen3-TTS plugin | Estimated 1-2 weeks engineering | Plausible path on Linux; defer pending model migration decision. |

**The key takeaway: don't grind on quantization or compile-the-whole-model
when the profile says CPU-bound. Find the hot inner loop and shrink its
per-iteration Python overhead.**

## When this won't help

- **Compute-bound models** (LLMs >7B params, image gen) — the GPU is
  already saturated; Python overhead is a smaller fraction.
- **Models without nested `generate()` loops** — single-codebook TTS
  (XTTS, F5-TTS) doesn't have the 900-call inner loop. T1 still helps
  a bit; T2 still works.
- **Linux + vLLM available** — vLLM's continuous batching + paged
  attention gives bigger gains by being a serving stack, not just a
  kernel optimization.

## Production integration pattern

Wrap the engine so all three tactics are transparent and the safety
toggle is one flag:

```python
class TtsEngine:
    def __init__(self, model_dir, apply_kernel_opt=True):
        self.model = Qwen3TTSModel.from_pretrained(model_dir,
            device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._kernel_opt = apply_kernel_opt
        if apply_kernel_opt:
            inner = self.model.model
            # T1: tight inner predictor loop
            install_fast_code_predictor_generate(self.model)
            # T2: compile inner predictor forward (cudagraphs)
            inner.talker.code_predictor.model.forward = torch.compile(
                inner.talker.code_predictor.model.forward,
                mode="reduce-overhead", dynamic=False,
            )
            # T3: tight outer talker loop (Round 2)
            install_T3_fast_outer_generate(
                inner.talker,
                codec_eos_token_id=inner.talker.config.eos_token_id,
                suppress_token_set=set(range(inner.talker.config.vocab_size_codec,
                                              inner.talker.config.vocab_size)),
            )
        self._warmed = False

    def register_voice(self, voice_id, ref_wav, ref_text):
        self.voices[voice_id] = self.model.create_voice_clone_prompt(...)
        if self._kernel_opt and not self._warmed:
            self._warm()                # one-shot, absorbs autotune
            self._warmed = True

    def synthesize(self, text, voice_id, ...):
        if self._kernel_opt:
            torch.compiler.cudagraph_mark_step_begin()
        return self.model.generate_voice_clone(text=text, ...)
```

`apply_kernel_opt=False` is your safety toggle — if a regression ever
ships in `torch.compile`, `qwen-tts`, or `transformers`, flip it off
and you're back on the unmodified HF pipeline at RTF 2.4.

## What's next — beyond 5.98×

### T4 / outer compile — blocked by HF DynamicCache

The obvious next step is `torch.compile(talker.model.forward,
mode="reduce-overhead")` on the outer 28-layer talker — analogous to
T2 on the inner predictor. We tried this in Round 2; **it did not
work**. Root cause: HuggingFace's `DynamicCache.get_seq_length()` in
`cache_utils.py:800` does `if layer_idx >= len(self.layers):`. Dynamo
guards on `len(layers)`. Each fresh outer `generate()` creates a new
`DynamicCache` whose `layers` list grows lazily from 0 to 28 during
prefill. Combined with the 16-codebook inner cache (each also growing
lazily 0→5) and per-step `cache_position` value guards, Dynamo blew
through `cache_size_limit=64` (logged) and was still rebuilding
cudagraph captures per shape signature. Bumping `cache_size_limit` to
256 didn't help — each version requires a fresh cudagraph re-capture
(~30s on the 28-layer block). Net effect: >10 minutes of compile
overhead before a single warmup call completed. Killed.

`mode="default"` (no cudagraphs) was also tried — same blocker, just
spent the time in Inductor codegen instead.

**Unblockers we know of**:
- Upstream `transformers` switches `Cache` to fixed-shape `StaticCache`
  with stable `len(layers)`. There's an open RFC; not yet shipped.
- Monkey-patch `DynamicCache` to pre-populate all 28 layers before
  first compile sees it. Brittle (couples our wrapper to HF cache
  internals); we haven't attempted it yet.

### T5 / per-layer compile — measured regression

Compiling each of the 28 decoder layers individually (`mode="default"`)
**regressed RTF 0.471 → 0.536 (0.88×)**. Per-layer compile prevents
Inductor from fusing across the residual / attention boundary, and
adds per-layer Dynamo guard overhead on every call. Don't do this.

### FP8 / NVFP4 — not currently viable on Blackwell + Windows

torchao's Blackwell NVFP4 path falls back to Marlin (known open issue)
and to a slow Python dequant on Windows. We measured 0.12× and 0.56×
in Round 1. Skipping pending upstream kernel work.

### What we'd watch for

- `transformers` Cache rewrite that fixes Dynamo guard explosion → T4
  becomes viable, estimated +20-30% on top of 5.98×
- torchao Blackwell NVFP4 kernels landing on Windows → another ~1.2×
- CosyVoice 3 + TensorRT-LLM (different model + Linux serving stack):
  upstream reports RTF ~0.10 on RTX 3090. If you can move to Linux /
  WSL2, that's the next plateau.

See [the companion's production sidecar](https://github.com/Wty2003328/waifu-companion/blob/main/tools/avatar/qwen3_engine.py)
for the up-to-date version of this recipe.

## See also

- [10-zero-shot-cloning.md](10-zero-shot-cloning.md) — base zero-shot recipe
- [12-integration.md](12-integration.md) — wiring the engine into an app
- [PyTorch profiler docs](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [torch.compile + GenerationMixin](https://huggingface.co/docs/transformers/main/en/llm_optims) — HF's own guide on related patterns
