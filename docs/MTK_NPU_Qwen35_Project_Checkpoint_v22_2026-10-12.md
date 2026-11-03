# Project checkpoint — Qwen3.5-4B on MediaTek MDLA (v22 prepared)

Updated 2026-10-12. Current known verified state is the user's successful v21 output-stage test. v22 extends that hidden vector with the block-1 FFN; v22 itself has not yet run on the phone.

## Device and paths
- POCO X7 Pro, Android 16, MT6899, NeuroPilot `libneuronusdk_adapter.mtk.so` 8.2.26, no root.
- Windows project root: `C:\Users\alikK\mtk-npu-project`.
- NDK clang: `C:\Android\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\clang.exe`.
- Model remains on phone: `/storage/emulated/0/models/Qwen3.5-4B-Q5_K_M.gguf`, size `3,143,656,608` bytes.
- Token selected in probes: `<|im_start|>` ID 248045. The GGUF inventory has no configured BOS token id.
- v10.1 inventory must be at `mtk_qwen35_gguf_audit_v10_1_adb_prefix\qwen35_gguf_inventory.json`.

## Verified real-weight path as of user v21 log
- v11.1: real `blk.0.attn_qkv.weight` (Q6_K -> FP16) used in MDLA FC.
- v12.1: real selected token embedding ID 248045 + real `blk.0.attn_norm.weight` + real QKV matrix on MDLA.
- v13: real block-0 Conv1D kernel + SiLU/QKV split, zero history (first token).
- v14/v15: real block-0 alpha/beta + `ssm_a`/dt-bias and one GDR step.
- v16/v17: block-0 linear-attention output, residual and FFN; output `/data/local/tmp/qwen35-v17-real-block0-hidden-token-0.bin` (MTKHID1 + 2560 FP16).
- v18: real block-1 `attn_norm` and QKV projection using v17 hidden.
- v19: real block-1 `ssm_conv1d` + SiLU/QKV split.
- v20: real block-1 alpha/beta, `ssm_a`, dt_bias and GDR.
- v21, confirmed by user's log: real block-1 `attn_gate`, `ssm_norm`, `ssm_out`, and residual add. It exported `/data/local/tmp/qwen35-v21-real-block1-post-attn-hidden-token-0.bin` (expected 5144 bytes: MTKHID1 header + 2560 FP16 values).
  - Gate FC max/mean error: 0.00097656 / 0.00000050.
  - Output gated RMSNorm max error: 0.00024414; out projection max error 0.00003052; residual sum max error 0.00006104.
  - Block-1 FFN was not included in v21.

## v22 prepared, not device-tested yet
Path `/mnt/data/mtk_qwen35_real_block1_ffn_probe_v22` / Windows sibling folder `mtk_qwen35_real_block1_ffn_probe_v22`.
- Reads v21 hidden input at `/data/local/tmp/qwen35-v21-real-block1-post-attn-hidden-token-0.bin`.
- Dynamically finds and extracts just four block-1 tensors from the on-phone GGUF using the v10.1 inventory: `blk.1.post_attention_norm.weight` F32, `blk.1.ffn_gate.weight` Q5_K, `blk.1.ffn_up.weight` Q5_K, `blk.1.ffn_down.weight` Q6_K.
- Casts/dequantizes all weights to FP16.
- Stage 1 computes real post-attention RMSNorm + gate/up FC projections on MDLA and exports `/data/local/tmp/qwen35-v22-block1-ffn-gate-up-token-0.bin`.
- Stage 2 computes SiLU(gate)*up, real down projection, residual ADD with the v21 input, outputs `/data/local/tmp/qwen35-v22-real-block1-post-ffn-hidden-token-0.bin` (MTKHID1 + [1,2560] FP16).
- Runner: `run_real_qwen35_block1_ffn_probe_v22.ps1`.
- Only syntax/packaging validation to be run here; Android build/runtime must be confirmed by user's phone log.

## Next if v22 succeeds
Block 1 post-FFN output is the input for block 2 (another linear-attention block). To proceed, obtain block-2 attention norm/QKV/conv/alpha-beta/ssm_a/dt_bias/gate/norm/out tensors and run the existing verified path with block-2 indexes and output names, then block-2 FFN. Block 3 is full-attention and requires actual full-attention semantics/rotary embedding/KV cache, not the GDR path.

## Key limitations
- Current path uses dequantized FP16 weights, not native Q5_K/Q6_K/Q8_0 NPU execution. This is not bit-exact GGUF/llama.cpp inference.
- Current selection is a single `<|im_start|>` token; real autoregressive tokenization and generation are not implemented.
- Entire model is not yet running end-to-end; future blocks, all 8 full-attention layers and KV cache, final norm, LM head, logits/sampling remain.
- No root/unlock/firmware changes should be required.
