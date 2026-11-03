/* Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT */

#ifndef NPU_TREE_VERIFIER_V21_H
#define NPU_TREE_VERIFIER_V21_H

#include <stdint.h>
#include <stddef.h>

#ifndef HIDDEN_DIM
#define HIDDEN_DIM 2560
#endif

#ifndef FFN_DIM
#define FFN_DIM 9216
#endif

#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM 256
#endif

#ifndef FA_KV_HEADS
#define FA_KV_HEADS 4
#endif

#ifndef FA_Q_PER_KV
#define FA_Q_PER_KV 4
#endif

#ifndef FA_Q_HEADS
#define FA_Q_HEADS (FA_KV_HEADS * FA_Q_PER_KV)
#endif

#ifndef NUM_BLOCKS
#define NUM_BLOCKS 32
#endif

#ifndef NUM_LIN_BLOCKS
#define NUM_LIN_BLOCKS 24
#endif

#ifndef NUM_FA_BLOCKS
#define NUM_FA_BLOCKS 8
#endif

#ifndef MAX_TREE_NODES
#define MAX_TREE_NODES 16
#endif

#ifndef MAX_TREE_DEPTH
#define MAX_TREE_DEPTH 6
#endif

#ifndef MAX_TREE_PATHS
#define MAX_TREE_PATHS 16
#endif

#ifndef CANDIDATE_TREE_DEFINED
#define CANDIDATE_TREE_DEFINED

typedef struct {
    int32_t  node_id;        // 0 .. MAX_TREE_NODES - 1
    uint32_t token_id;       // Proposed token ID (0 .. VOCAB_SIZE-1)
    int32_t  parent_idx;     // Parent node index in tree, -1 for root
    int32_t  depth;          // Depth in tree (1 .. MAX_TREE_DEPTH)
    int32_t  seq_pos_offset; // Relative position offset (depth = 1..D)
    float    confidence;     // Heuristic probability score (0.0 .. 1.0)
    uint8_t  source_tier;    // TIER_*
    uint8_t  is_leaf;        // 1 if no children in this tree
} CandidateTreeNode;

typedef struct {
    int length;
    int node_indices[MAX_TREE_DEPTH];
    float path_confidence;
} TreePath;

typedef struct {
    int num_nodes;           // Number of candidate nodes (8 .. 15)
    int max_depth;           // Maximum depth reached (4 .. 6)
    uint32_t root_token;     // Last verified token t_0
    int base_seq_pos;        // Base sequence position
    CandidateTreeNode nodes[MAX_TREE_NODES];
    uint16_t ancestry_mask[MAX_TREE_NODES];
    int num_paths;
    TreePath paths[MAX_TREE_PATHS];
    float overall_confidence;
    int decision;
} CandidateTree;

#endif // CANDIDATE_TREE_DEFINED

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAX_KV_SEQ
#define MAX_KV_SEQ 2048
#endif

// Generates 4D Tree Ancestry Mask for B=16 and dynamic sequence length S (80, 128, 640, 2048)
void build_4d_tree_mask_fp16(
    const CandidateTree *tree,
    int base_pos,
    int batch_len,
    int S,
    __fp16 *mask_out);

// Compiles and verifies the MediaTek NPU 880 batched tree verification graphs:
// 1. tree_gqa_B16_S128, S640, S2048 (Tree-Attention GQA with 4D ancestry mask on mtk-mdla)
// 2. universal_dynamic_ffn_B16 (Universal Dynamic FFN SwiGLU servicing all 32 blocks)
// Returns 1 on success, 0 on failure.
int init_npu_tree_verification_graphs(void);

// Compiles a zero-DMA resident Hardware INT4 FFN graph on MDLA for a specific block (0..31):
// Uses native B=4 batch dimension, TENSOR_QUANT8_SYMM_PER_CHANNEL weights, TENSOR_QUANT16_SYMM activations,
// and --opt=3 --fc-to-conv --split-16a4w-conv-oc --use-int4-wt --mdla-conv-exp.
int npu_compile_pinned_int4_block_ffn(
    int blk,
    const __fp16 *w_gate,
    const __fp16 *w_up,
    const __fp16 *w_down);

// Compiles zero-DMA resident Pinned FFN graphs for all blocks:
int npu_compile_pinned_fa_ffn_graphs(void);

// Registers or updates dynamic weights for a specific block (0..31):
void npu_set_block_ffn_weights(int blk, const __fp16 *w_gate, const __fp16 *w_up, const __fp16 *w_down, const __fp16 *gamma);

// Benchmarks the Universal Dynamic FFN graph latency on device for B=16:
int benchmark_universal_ffn_latency(int iters, double *out_mean_ms, double *out_gflops);

// Executes Hardware INT4 FFN SwiGLU on mtk-mdla in a SINGLE SHOT for batch_len (1..4) tokens:
// - x_in: [batch_len, 2560]
// - gamma: [2560] post_norm RMSNorm weights (optional if already normalized)
// - x_out: [batch_len, 2560] computed residual output (x + Down)
// Returns 1 on success, 0 on failure (falls back to CPU SDOT).
int run_real_npu_universal_ffn(
    int blk,
    int batch_len,
    const __fp16 (*x_in)[HIDDEN_DIM],
    const __fp16 *gamma,
    __fp16 (*x_out)[HIDDEN_DIM],
    double *out_ffn_ms);

// Executes real batched Tree-GQA directly on mtk-mdla using real FP16 activations and KV-cache:
int run_real_npu_tree_gqa(
    int full_idx,
    int base_pos,
    int batch_len,
    const __fp16 *q_act,
    const __fp16 *gate_act,
    const void *k_cache,
    const void *v_cache,
    const CandidateTree *tree,
    __fp16 *attn_out,
    double *out_gqa_ms);

// Executes a candidate tree verification block on NPU 880 (MDLA 5.x):
int run_npu_batched_tree_block(
    int blk,
    int is_full,
    int batch_len,
    int base_pos,
    __fp16 (*batch_hidden)[HIDDEN_DIM],
    const CandidateTree *tree,
    double *out_sdot_ms,
    double *out_npu_ms);

#ifdef __cplusplus
}
#endif

#endif // NPU_TREE_VERIFIER_V21_H
