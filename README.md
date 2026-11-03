# NPU 880 — Experimental LLM Runtime for MediaTek NPU (MT6899 / Dimensity 8400 Ultra)

**Status:** Work in Progress — research code, not production-ready.
**License:** MIT (see LICENSE) — original code only; see DISCLAIMER.md.
**Test device:** POCO X7 Pro (MediaTek Dimensity 8400 Ultra, MT6899, NPU 880), Android 15 / HyperOS 2.
**Target model:** Qwen3.5-4B, GGUF Q4_K_M (~2.7 GB, not included).

## What it is
An independent C inference engine that runs an autoregressive LLM on the phone's NPU
via the on-device NeuroPilot/MDLA adapter (`dlopen` + `dlsym` at runtime — no vendor files
bundled), with NEON CPU fallback paths:

- own GGUF v3 parser + byte-level BPE tokenizer/detokenizer
- Q4_K/Q5_K/Q6_K/Q8_0 dequant + ARM NEON SDOT/dotprod/i8mm GEMM kernels (8-thread pinned pool)
- NPU graph construction/compilation/execution through the runtime adapter API
- fused per-block NPU graphs, zero-recompile multi-prompt sessions, pinned DMA/ION slabs
- speculative decoding: adaptive n-gram/PLD/syntax tree drafter + lossless tree verifier
  with exact KV-cache commit/rollback
- experimental HTTP/SSE inference server

## Layout
```
src/            r22 engine sources (v182_r22_breakthrough_engine.c, adaptive_tree_drafter_v22.*,
                npu_tree_verifier_v22.*)
build_r22.py    on-device build pipeline (Termux/clang, -mcpu=cortex-a725, 16K pages)
run_termux_r22.sh  Termux launcher
docs/           project checkpoints & notes
```

## Build & run (on the device, Termux)
```bash
python3 build_r22.py                 # or TERMUX_COPY_DIR=... to also copy into Termux
./run_termux_r22.sh "Привет! ..." 128
# model path via GGUF_PATH, binary via BIN_PATH env vars
```

## Current performance (WIP, measured on the test device)
- M=1 decode ~105 ms/tok (~9.5 tok/s) base; speculative batch (B<=8) raises effective
  tok/s with the accepted-draft count. Target: 50-70 tok/s — see docs/.
- Known bottlenecks are documented in docs/ (LM head, ioctl turnaround, FP16 weight traffic).

## Disclaimer
Independent research project — **not affiliated with MediaTek**. No proprietary MediaTek
files are included; the binary only uses system libraries already present on the device.
See DISCLAIMER.md.
