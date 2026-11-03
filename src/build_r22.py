# Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
build_r22.py - Build pipeline for Qwen3.5-4B v182-r22
Target: POCO X7 Pro = MediaTek Dimensity 8400 Ultra (MT6899) / NPU 880, 8x Cortex-A725 (ARMv9.2-A)

r22 changes vs r21:
 1. -mcpu=cortex-a725 alone (implies armv9.2-a + dotprod + i8mm + fp16); removed the redundant
    -march=armv8.6-a+... that conflicted with -mcpu (clang warning).
 2. -Wl,-z,max-page-size=16384  -> binary loads on Android 15 devices with 16 KB kernel pages.
 3. -ffp-contract=fast.
 4. Optional copy of the binary into the Termux working dir (set TERMUX_COPY_DIR env),
    so run_termux_r22.sh finds it without manual copying.
 5. Draft budget raised to 7 (see adaptive_tree_drafter_v22.h SPEC_DRAFT_BUDGET).
"""

import os
import sys
import shutil
import subprocess
import time

# [r22 FIX] Resolve everything relative to THIS script's location (the repo's src/ dir),
# instead of a hard-coded /data/user/0/com.agy/files/qwen35_npu path.
# Override with env var if you keep sources elsewhere:
#   QWEN35_NPU_DIR=/path/to/dir python3 build_r22.py
BASE_DIR = os.environ.get(
    "QWEN35_NPU_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
R22_SRC = os.path.join(BASE_DIR, "v182_r22_breakthrough_engine.c")
VERIFIER_SRC = os.path.join(BASE_DIR, "npu_tree_verifier_v22.c")
DRAFTER_SRC = os.path.join(BASE_DIR, "adaptive_tree_drafter_v22.c")
BIN_OUT = os.path.join(BASE_DIR, "v182_r22_bin")

# Optional: copy result next to run_termux_r22.sh in Termux, e.g.
#   TERMUX_COPY_DIR=/data/data/com.termux/files/home/qwen35_npu python3 build_r22.py
TERMUX_COPY_DIR = os.environ.get("TERMUX_COPY_DIR", "")

print("================================================================================")
print("  MediaTek MDLA INT4 Engine build v182-r22 (Dimensity 8400 Ultra / NPU 880)        ")
print("================================================================================\n")

print("[Step 1/3] Verifying source files...")
for src in [R22_SRC, VERIFIER_SRC, DRAFTER_SRC]:
    if not os.path.exists(src):
        print(f"[FATAL ERROR] Required source file missing: {src}")
        sys.exit(1)
    print(f"  -> Found {os.path.basename(src)} ({os.path.getsize(src):,} bytes)")

print("\n[Step 2/3] Compiling v182_r22_bin with clang...")
# NOTE: -mcpu=cortex-a725 implies ARMv9.2-A + dotprod + i8mm + fp16 on this core,
# so no separate -march is needed (mixing both makes clang drop one of them with a warning).
compile_cmd = [
    "clang", "-O3",
    "-mcpu=cortex-a725",
    "-ffp-contract=fast",
    "-Wl,-z,max-page-size=16384",   # required on 16 KB-page kernels (Android 15)
    "-std=gnu11",
    "-pthread",
    "-o", BIN_OUT,
    R22_SRC,
    DRAFTER_SRC,
    VERIFIER_SRC,
    "-ldl", "-lm"
]

t0 = time.time()
res = subprocess.run(compile_cmd, cwd=BASE_DIR, capture_output=True, text=True)
t_comp = time.time() - t0

if res.returncode != 0:
    print("[ERROR] Compilation failed!")
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    sys.exit(1)
if res.stderr.strip():
    print("[clang warnings]\n" + res.stderr.strip()[:4000])

os.chmod(BIN_OUT, 0o755)
print(f"  -> OK {BIN_OUT} ({os.path.getsize(BIN_OUT):,} bytes) in {t_comp:.2f} s")

print("\n[Step 3/3] Sanity check + optional Termux copy...")
res_file = subprocess.run(["file", BIN_OUT], capture_output=True, text=True)
print("  ->", res_file.stdout.strip())

if TERMUX_COPY_DIR:
    os.makedirs(TERMUX_COPY_DIR, exist_ok=True)
    dst = os.path.join(TERMUX_COPY_DIR, "v182_r22_bin")
    shutil.copy2(BIN_OUT, dst)
    os.chmod(dst, 0o755)
    print(f"  -> Copied to {dst}")

print("\n================================================================================")
print("  v182_r22_bin BUILD COMPLETE                                                   ")
print("================================================================================\n")
