# Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT

#!/data/data/com.termux/files/usr/bin/bash

# ==============================================================================
# Qwen3.5-4B NPU Engine (v182-r22) — Termux Launcher
# Target: POCO X7 Pro (MediaTek Dimensity 8400 Ultra / MT6899 / NPU 880)
# ==============================================================================

DIR="$(cd "$(dirname "$0")" && pwd)"

# [r22] Binary lookup order:
#   1. $BIN_PATH (explicit override)
#   2. next to this script
#   3. $HOME/qwen35_npu  (typical Termux working dir)
#   4. the com.agy build dir used by build_r22.py (when run on-device with access)
BIN=""
for cand in \
    "${BIN_PATH:-}" \
    "$DIR/v182_r22_bin" \
    "$HOME/qwen35_npu/v182_r22_bin" \
    "/data/user/0/com.agy/files/qwen35_npu/v182_r22_bin"
do
    if [ -n "$cand" ] && [ -f "$cand" ]; then BIN="$cand"; break; fi
done

if [ -z "$BIN" ]; then
    echo "[ERROR] v182_r22_bin not found in any of:"
    echo "        \$BIN_PATH, $DIR, \$HOME/qwen35_npu, /data/user/0/com.agy/files/qwen35_npu"
    echo "[HINT] Build it first, from this directory:"
    echo "        pkg install clang python   # if not installed yet"
    echo "        python3 build_r22.py       # produces ./v182_r22_bin next to this script"
    echo "  or: BIN_PATH=/path/to/v182_r22_bin $0 ..."
    exit 1
fi

chmod +x "$BIN" 2>/dev/null

# Shared storage (/storage/emulated/0) is mounted NOEXEC on Android:
# executing a binary from there always fails with "Permission denied".
case "$BIN" in
    /storage/emulated/0/*|/sdcard/*)
        echo "[ERROR] Binary is on shared storage (noexec): $BIN"
        echo "[HINT] Copy the repo into Termux private storage and run from there:"
        echo "        mkdir -p ~/npu880 && cp -r /storage/emulated/0/npu880/* ~/npu880/"
        echo "        cd ~/npu880/src && python3 build_r22.py && chmod +x run_termux_r22.sh v182_r22_bin"
        echo "        ./run_termux_r22.sh \"Привет!\" 64"
        exit 1
        ;;
esac

DEFAULT_MODEL="/storage/emulated/0/Файлы проекта/models/Qwen3.5-4B-Q4_K_M.gguf"
MODEL="${GGUF_PATH:-$DEFAULT_MODEL}"
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] Model file not found: $MODEL"
    echo "[HINT] Put the GGUF model there, or set GGUF_PATH, e.g.:"
    echo "        GGUF_PATH=/storage/emulated/0/Download/Qwen3.5-4B-Q4_K_M.gguf $0 \"Привет!\" 64"
    exit 1
fi
PROMPT="${1:-Привет! Расскажи подробно, как устроена Солнечная система.}"
MAX_TOKENS="${2:-128}"

echo "=================================================================="
echo "  Starting Qwen3.5-4B Hardware INT4 MDLA Engine (v182-r22)"
echo "  Bin:    $BIN"
echo "  Model:  $MODEL"
echo "  Prompt: $PROMPT"
echo "  Tokens: $MAX_TOKENS"
echo "=================================================================="

# ВАЖНО: env -u LD_PRELOAD -u LD_LIBRARY_PATH необходим в Termux,
# чтобы Termux не перехватывал dlopen системных библиотек MediaTek NPU (/vendor/lib64/*).
# Если бинарь при этом не стартует ("library not found"), пересоберите его статически
# (libc++ в Termux по умолчанию статический) — проверьте: ldd "$BIN"
exec env -u LD_PRELOAD -u LD_LIBRARY_PATH \
  "$BIN" \
  -gguf "$MODEL" \
  -prompt "$PROMPT" \
  -maxtokens "$MAX_TOKENS"
