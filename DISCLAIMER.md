# Disclaimer

This project is an **independent student research project**. It is **not affiliated with,
endorsed by, or sponsored by MediaTek Inc.** or Xiaomi/POCO. "MediaTek", "Dimensity",
"NeuroPilot", "NPU" and related names are trademarks of their respective owners and are
used here only to describe the hardware/software the project runs on.

## What this repository contains
- Original, independently written C source code (inference engine, GGUF parser, tokenizer,
  speculative decoding, build scripts).
- Original documentation and development notes.

## What this repository does NOT contain
- No MediaTek SDK files, headers, vendor libraries, firmware, blobs or confidential
  documentation are included or redistributed.
- The compiled program does **not** bundle any vendor binary. At runtime it only calls
  (`dlopen`/`dlsym`) system libraries that are **already present on the user's own device**
  (e.g. the on-device NeuroPilot adapter library) — the same way any Android app uses
  system APIs.

## Reverse-engineered interface notes
- Function signatures of the on-device adapter are declared in our own words as
  function-pointer typedefs and resolved at runtime with `dlsym`. No vendor header text
  is copied.
- `ANeuralNetworks`-style operand structures follow the public AOSP NNAPI specification,
  which is licensed under the **Apache License 2.0** (© The Android Open Source Project).
- The single DMA-heap `ioctl` request structure is a minimal 4-field ABI re-declaration
  of a publicly observable kernel interface (facts/ABI are not copyrightable; we keep the
  expression minimal and original).

## No warranty
This is experimental work-in-progress research code. It may be unstable, incomplete or
incorrect. Use at your own risk.
