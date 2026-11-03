NPU 880 — Experimental LLM Runtime for MediaTek NPU (MT6899 / Dimensity 8400 Ultra)

    ⚠️ KNOWN ISSUE — CURRENT BUILD IS BUGGY: the r22 build compiles and runs on-device
    (all 32 NPU graphs compile, prefill works), but generated text is currently unreliable —
    on many prompts it degenerates into repetition/garbage after some context length.
    The speculative-decoding state management (tree verify + KV-cache commit/rollback + GDR
    replay) is under active debugging. Treat generation output as broken until fixed.
    Bisect with the -nospec / -purenpu / -npugdr flags (see
    run_termux_r22.sh).

Status: Work in Progress — research code, not production-ready.
License: MIT (see LICENSE) — original code only; see
DISCLAIMER.md.
Test device: POCO X7 Pro (MediaTek Dimensity 8400 Ultra, MT6899, NPU 880), Android 16 / HyperOS 3.
Target model: Qwen3.5-4B, GGUF Q4_K_M (~2.7 GB, not included).
What it is

An independent C inference engine that runs an autoregressive LLM on the phone's NPU
via the on-device NeuroPilot/MDLA adapter (dlopen + dlsym at runtime — no vendor files
bundled), with NEON CPU fallback paths:

    own GGUF v3 parser + byte-level BPE tokenizer/detokenizer
    Q4_K/Q5_K/Q6_K/Q8_0 dequant + ARM NEON SDOT/dotprod/i8mm GEMM kernels (8-thread pinned pool)
    NPU graph construction/compilation/execution through the runtime adapter API
    fused per-block NPU graphs, zero-recompile multi-prompt sessions, pinned DMA/ION slabs
    speculative decoding: adaptive n-gram/PLD/syntax tree drafter + lossless tree verifier
    with exact KV-cache commit/rollback (currently buggy — see warning above)
    experimental HTTP/SSE inference server

Layout

text

src/            r22 engine sources: v182_r22_breakthrough_engine.c, adaptive_tree_drafter_v22.*,
                npu_tree_verifier_v22.*, build_r22.py (on-device build), run_termux_r22.sh (launcher)
docs/           project checkpoints & notes

Build & run (on the device, Termux)

Bash

cd src
pkg install clang python        # if not installed
python3 build_r22.py            # builds ./v182_r22_bin next to the sources
chmod +x run_termux_r22.sh v182_r22_bin
termux-setup-storage            # access to the model on /storage
GGUF_PATH=/storage/emulated/0/.../Qwen3.5-4B-Q4_K_M.gguf \
  ./run_termux_r22.sh "Привет! Расскажи подробно, как устроена Солнечная система." 128
# extra engine flags are forwarded, e.g.:  ./run_termux_r22.sh "..." 64 -nospec
# binary lookup: $BIN_PATH -> script dir -> $HOME/qwen35_npu

    Note: executables must live in Termux private storage ($HOME), NOT on
    /storage/emulated/0 — shared storage is mounted noexec on Android.

Measured so far (WIP, test device)

    NPU graph compile: 32/32 INT4 FFN graphs OK (~54 s total)
    Prefill: ~9 tok/s (110 ms/tok at 14-token prompt)
    Decode: ~4–5.5 tok/s on short prompts (target: 50–70 tok/s)
    Known bottlenecks: CPU SDOT share per step (~100–135 ms), NPU block time (~110 ms/tok),
    LM head, ioctl turnaround, FP16 weight traffic (native INT4 = next step)

Disclaimer

Independent research project — not affiliated with MediaTek. No proprietary MediaTek
files are included; the binary only uses system libraries already present on the device.
See
DISCLAIMER.md.
