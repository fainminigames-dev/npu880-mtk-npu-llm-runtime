/* Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT */

// =============================================================================
// adaptive_tree_drafter.h - High-Performance Adaptive Speculative Candidate
// Tree Drafting, Verification, and KV-Cache State Management Engine for
// Qwen3.5-4B on ARMv9 Cortex-A725 (MediaTek Dimensity 8400)
// =============================================================================
#ifndef ADAPTIVE_TREE_DRAFTER_V21_H
#define ADAPTIVE_TREE_DRAFTER_V21_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------------
// Sizing and Architectural Constants
// -----------------------------------------------------------------------------
#ifndef VOCAB_SIZE
#define VOCAB_SIZE              248320
#endif

#ifndef MAX_CONTEXT_LEN
#define MAX_CONTEXT_LEN         4096
#endif

#ifndef MAX_TREE_NODES
#define MAX_TREE_NODES          16
#endif

#ifndef MAX_TREE_DEPTH
#define MAX_TREE_DEPTH          8   // [r22] raised 6 -> 8: allows up to 8-deep linear draft trunk
#endif

// [r22] Linear speculative trunk length (draft candidates per step). r21 used a hard-coded 3,
// capping throughput at 4 tokens/step. 7 gives batch M = 1..8 (<= SPEC_MAX_BATCH = 16 in engine).
// Raise toward 15 only together with engine SPEC_MAX_DRAFT and measured acceptance rate.
#ifndef SPEC_DRAFT_BUDGET
#define SPEC_DRAFT_BUDGET       7
#endif

#ifndef MAX_TREE_PATHS
#define MAX_TREE_PATHS          16
#endif

#ifndef NUM_LINEAR_BLOCKS
#define NUM_LINEAR_BLOCKS       24
#endif

#ifndef NUM_FULL_BLOCKS
#define NUM_FULL_BLOCKS         8
#endif

#ifndef TOTAL_MODEL_BLOCKS
#define TOTAL_MODEL_BLOCKS      32
#endif

#ifndef FA_KV_HEADS
#define FA_KV_HEADS             4
#endif

#ifndef FA_Q_PER_KV
#define FA_Q_PER_KV             4
#endif

#ifndef FA_HEAD_DIM
#define FA_HEAD_DIM             256
#endif

#ifndef MAX_KV_SEQ
#define MAX_KV_SEQ              2048
#endif

#ifndef QKV_DIM
#define QKV_DIM                 8192
#endif

#ifndef CONV_KW
#define CONV_KW                 4
#endif

#ifndef NVALUE
#define NVALUE                  32
#endif

#ifndef GDR_STATE_ELEMS
#define GDR_STATE_ELEMS         16384 // 128 x 128 __fp16
#endif

// In-Cache Table Dimensions
#define PLD_HASH_BUCKETS        65536
#define NGRAM_CACHE_SETS        32768
#define NGRAM_CACHE_WAYS        4
#define BPE_PREFIX_HASH_SIZE    16384
#define BPE_MAX_COMPLETION      6
#define SYNTAX_MAX_BRANCHES     4
#define SYNTAX_RULE_HASH_SIZE   4096

#define CONFIDENCE_HIGH_THRESHOLD   0.25f
#define CONFIDENCE_PATH_MIN         0.20f

// Source Tiers
#define TIER_PLD_5GRAM          1
#define TIER_PLD_4GRAM          2
#define TIER_PLD_3GRAM          3
#define TIER_PLD_2GRAM          4
#define TIER_NGRAM_4WAY         5
#define TIER_BPE_PREFIX         6
#define TIER_SYNTACTIC          7

// Proposal Mode Decision
typedef enum {
    TREE_PROPOSAL_M1_FAST   = 0, // Low confidence / novel: FAST M=1 mode (105 ms = 9.5 tok/s, zero penalty)
    TREE_PROPOSAL_SPEC_TREE = 1  // High confidence (>0.60): Speculative candidate tree (8-16 candidates)
} AdaptiveProposalDecision;

// -----------------------------------------------------------------------------
// Candidate Tree Node and Path Structures (Protected by CANDIDATE_TREE_DEFINED)
// -----------------------------------------------------------------------------
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
    int num_nodes;           // Number of candidate nodes (0 for M=1 fallback, or 8 .. 16)
    int max_depth;           // Maximum depth reached (1 .. 6)
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

// -----------------------------------------------------------------------------
// Lossless Tree Verification Result
// -----------------------------------------------------------------------------
typedef struct {
    int num_accepted_drafts;              // Number of accepted speculative tokens (0 .. D)
    int total_emitted_tokens;             // 1 (bonus token) + accepted_drafts
    uint32_t emitted_tokens[MAX_TREE_DEPTH + 1];
    int winning_path_idx;
    int winning_leaf_node_id;
    int winning_node_ids[MAX_TREE_DEPTH]; // Exact node IDs along the accepted prefix
} TreeVerificationResult;

// -----------------------------------------------------------------------------
// Component A: In-Cache Prompt Lookup Decoding (PLD) Suffix Index
// -----------------------------------------------------------------------------
typedef struct {
    int32_t head[PLD_HASH_BUCKETS];
    int32_t prev_occ[MAX_CONTEXT_LEN];
    uint32_t tokens[MAX_CONTEXT_LEN];
    int length;
    int prompt_len;
} PldIndex;

typedef struct {
    uint32_t continuation[MAX_TREE_DEPTH];
    int length;
    int match_order; // 5, 4, 3, 2
    int is_from_prompt;
    float confidence;
} PldMatchBranch;

// -----------------------------------------------------------------------------
// Component B: Dynamic 4-Way Set-Associative N-Gram Cache
// -----------------------------------------------------------------------------
typedef struct {
    uint64_t key_hash;
    uint32_t next_token;
    uint16_t freq;
    uint16_t step_lru;
    uint8_t  valid;
    uint8_t  order;
} NgramSlot4Way;

typedef struct {
    NgramSlot4Way slots[NGRAM_CACHE_WAYS];
} NgramSet4Way;

typedef struct {
    NgramSet4Way sets[NGRAM_CACHE_SETS];
    uint32_t cur_step;
    uint64_t total_lookups;
    uint64_t total_hits;
    uint64_t total_inserts;
} DynamicNgramEngine;

typedef struct {
    uint32_t token_id;
    uint16_t freq;
    uint8_t  order;
    float    score;
} NgramCandidate;

// -----------------------------------------------------------------------------
// Component C: Dynamic BPE Multi-Token Prefix Prediction Engine
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t stem_token;
    uint32_t completion[BPE_MAX_COMPLETION];
    uint8_t  comp_len;
    uint8_t  valid;
    float    confidence;
} BpePrefixEntry;

typedef struct {
    BpePrefixEntry table[BPE_PREFIX_HASH_SIZE];
} BpePrefixEngine;

// -----------------------------------------------------------------------------
// Component D: Syntactic & Collocation Tree Engine
// -----------------------------------------------------------------------------
typedef struct {
    uint32_t tokens[MAX_TREE_DEPTH];
    int      length;
    float    confidence;
} SyntacticBranch;

typedef struct {
    uint32_t trigger_token;
    uint32_t context_prev;  // 0xFFFFFFFFu if wildcard (matches any context)
    int      num_branches;
    SyntacticBranch branches[SYNTAX_MAX_BRANCHES];
    uint8_t  valid;
} SyntacticRule;

typedef struct {
    SyntacticRule table[SYNTAX_RULE_HASH_SIZE];
    uint32_t total_rules;
} SyntacticEngine;

// -----------------------------------------------------------------------------
// Aggregated Adaptive Tree Speculative Drafter Engine
// -----------------------------------------------------------------------------
typedef struct {
    PldIndex           pld;
    DynamicNgramEngine ng;
    BpePrefixEngine    bpe;
    SyntacticEngine    syntax;

    // Statistics
    uint64_t total_proposals;
    uint64_t total_tree_proposals;
    uint64_t total_m1_fallbacks;
    uint64_t total_accepted_drafts;
    uint64_t total_emitted_tokens;
} AdaptiveTreeEngine;

// Model state layout for exact zero-copy KV-cache commit
typedef struct {
    __fp16 conv_hist[NUM_LINEAR_BLOCKS][CONV_KW - 1][QKV_DIM];
    __fp16 gdr_state[NUM_LINEAR_BLOCKS][NVALUE][GDR_STATE_ELEMS];
    __fp16 k_cache[NUM_FULL_BLOCKS][FA_KV_HEADS][FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM];
    __fp16 v_cache[NUM_FULL_BLOCKS][FA_KV_HEADS][FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM];
} AdaptiveModelState;

// Forward declaration of model state
struct ModelState;

// -----------------------------------------------------------------------------
// Public Engine API
// -----------------------------------------------------------------------------

// Initialize the entire drafter engine, seeding canonical BPE, n-gram, and syntactic rule tables
void adaptive_tree_engine_init(AdaptiveTreeEngine *eng);

// Load prompt or context tokens into PLD and N-gram cache
void adaptive_tree_load_context(
    AdaptiveTreeEngine *eng,
    const uint32_t *tokens,
    int count,
    int is_prompt);

// Check if a token is a terminal stop / end-of-sequence token (<|im_end|>, etc.)
int adaptive_tree_is_stop_or_special(uint32_t token_id);

// Check if a token is a single punctuation mark, boundary, newline, or stop token (retained for backward compatibility)
int adaptive_tree_is_punct_or_boundary(uint32_t token_id);

// Computes tree path confidence and proposes an optimal candidate tree.
// Returns TREE_PROPOSAL_M1_FAST (0) if confidence is low (<=0.60) or at EOS,
// or TREE_PROPOSAL_SPEC_TREE (1) if confidence is high (>0.60) proposing 8-16 candidates.
AdaptiveProposalDecision adaptive_tree_propose(
    AdaptiveTreeEngine *eng,
    const uint32_t *history_ids,
    int hist_len,
    int base_seq_pos,
    int max_candidates,
    CandidateTree *out_tree);

// Lossless argmax verification against target model output predictions
void adaptive_tree_verify_lossless(
    const CandidateTree *tree,
    uint32_t target_root_prediction,
    const uint32_t *target_node_predictions,
    TreeVerificationResult *result);

// Exact KV-cache commit / rollback:
// If num_accepted_drafts == 0: leaves base_pos intact, no movement.
// If num_accepted_drafts > 0: moves winning branch KV from scratch slots to exact sequence positions.
void adaptive_tree_kv_cache_commit(
    void *model_state,
    int base_pos,
    const CandidateTree *tree,
    const TreeVerificationResult *verif);

// Extracts the winning path sequence of batch indices (p_0, p_1, ...) for GDR recurrent state replay
void adaptive_tree_extract_gdr_replay_path(
    const TreeVerificationResult *verif,
    int *out_winning_batch_indices,
    int *out_winning_count);

// Updates PLD and N-gram cache online with newly accepted tokens
void adaptive_tree_update_caches(
    AdaptiveTreeEngine *eng,
    const uint32_t *history_ids,
    int hist_len_before,
    const TreeVerificationResult *verif);

#ifdef __cplusplus
}
#endif

#endif // ADAPTIVE_TREE_DRAFTER_V21_H
