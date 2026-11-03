/* Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT */

// =============================================================================
// adaptive_tree_drafter.c - High-Performance Adaptive Speculative Candidate
// Tree Drafting, Verification, and KV-Cache State Management Engine for
// Qwen3.5-4B on ARMv9 Cortex-A725 (MediaTek Dimensity 8400)
// =============================================================================

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
// [r22 FIX] NEON header was included unconditionally although this file uses zero
// NEON intrinsics — it broke host/other-arch builds. Guard it:
#if defined(__ARM_NEON) && defined(__has_include)
#if __has_include(<arm_neon.h>)
#include <arm_neon.h>
#define DRAFTER_HAVE_NEON 1
#endif
#endif

#include "adaptive_tree_drafter_v22.h"

// -----------------------------------------------------------------------------
// Helper: Hash Functions
// -----------------------------------------------------------------------------
static inline uint32_t hash_token_pld(uint32_t token) {
    return (token * 2654435761u) & (PLD_HASH_BUCKETS - 1);
}

static inline uint64_t hash_ngram_context(uint32_t prev3, uint32_t prev2, uint32_t prev1) {
    uint64_t h = (uint64_t)prev1 * 2654435761ULL;
    if (prev2 != 0xFFFFFFFFu) h ^= ((uint64_t)prev2 * 3266489917ULL);
    if (prev3 != 0xFFFFFFFFu) h ^= ((uint64_t)prev3 * 2246822519ULL);
    return h;
}

static inline uint32_t ngram_set_index(uint64_t hash) {
    return (uint32_t)((hash ^ (hash >> 16)) & (NGRAM_CACHE_SETS - 1));
}

static inline uint32_t hash_syntax_context(uint32_t prev2, uint32_t prev1) {
    uint32_t h = prev1 * 2654435761u;
    if (prev2 != 0xFFFFFFFFu) h ^= (prev2 * 3266489917u);
    return h & (SYNTAX_RULE_HASH_SIZE - 1);
}

// -----------------------------------------------------------------------------
// Component A: In-Cache Prompt Lookup Decoding (PLD)
// -----------------------------------------------------------------------------
static void pld_index_init(PldIndex *idx) {
    memset(idx->head, -1, sizeof(idx->head));
    idx->length = 0;
    idx->prompt_len = 0;
}

static void pld_index_append(PldIndex *idx, uint32_t token) {
    if (idx->length >= MAX_CONTEXT_LEN) return;
    int pos = idx->length;
    idx->tokens[pos] = token;
    uint32_t b = hash_token_pld(token);
    idx->prev_occ[pos] = idx->head[b];
    idx->head[b] = pos;
    idx->length++;
}

static int pld_lookup_continuations(
    const PldIndex *idx,
    const uint32_t *suffix,
    int suffix_len,
    PldMatchBranch *out_branches,
    int max_branches)
{
    if (suffix_len < 1 || idx->length < 2 || max_branches < 1) return 0;
    int n_branches = 0;
    uint32_t last_tok = suffix[suffix_len - 1];
    uint32_t b = hash_token_pld(last_tok);

    for (int order = (suffix_len >= 4 ? 4 : suffix_len); order >= 1 && n_branches < max_branches; --order) {
        int req_match = order;
        int p = idx->head[b];
        while (p >= 0 && n_branches < max_branches) {
            if (p < idx->length - 1 && p >= req_match - 1 && p != idx->length - 1) {
                int match = 1;
                for (int m = 1; m < req_match; ++m) {
                    if (idx->tokens[p - m] != suffix[suffix_len - 1 - m]) {
                        match = 0;
                        break;
                    }
                }
                if (match) {
                    PldMatchBranch br;
                    br.match_order = req_match + 1;
                    br.is_from_prompt = (p < idx->prompt_len);
                    int copy_len = idx->length - 1 - p;
                    if (copy_len > MAX_TREE_DEPTH) copy_len = MAX_TREE_DEPTH;
                    for (int c = 0; c < copy_len; ++c) {
                        br.continuation[c] = idx->tokens[p + 1 + c];
                    }
                    br.length = copy_len;
                    br.confidence = 0.90f + 0.02f * (float)req_match + (br.is_from_prompt ? 0.05f : 0.0f);
                    if (br.confidence > 0.98f) br.confidence = 0.98f;

                    int dup = 0;
                    for (int eb = 0; eb < n_branches; ++eb) {
                        if (out_branches[eb].length > 0 && br.length > 0 &&
                            out_branches[eb].continuation[0] == br.continuation[0]) {
                            dup = 1;
                            break;
                        }
                    }
                    if (!dup && br.length > 0) {
                        out_branches[n_branches++] = br;
                    }
                }
            }
            p = idx->prev_occ[p];
        }
    }
    return n_branches;
}

// -----------------------------------------------------------------------------
// Component B: Dynamic 4-Way Set-Associative N-Gram Cache
// -----------------------------------------------------------------------------
static void ngram_engine_init(DynamicNgramEngine *ng) {
    memset(ng->sets, 0, sizeof(ng->sets));
    ng->cur_step = 0;
    ng->total_lookups = 0;
    ng->total_hits = 0;
    ng->total_inserts = 0;
}

static void ngram_engine_record(
    DynamicNgramEngine *ng,
    uint32_t prev3,
    uint32_t prev2,
    uint32_t prev1,
    uint32_t next_tok,
    uint8_t order)
{
    if (next_tok >= VOCAB_SIZE || next_tok == 248045 || next_tok == 248046 || next_tok == 248044) return;
    ng->total_inserts++;
    uint64_t kh = hash_ngram_context(prev3, prev2, prev1);
    uint32_t set_idx = ngram_set_index(kh);
    NgramSet4Way *set = &ng->sets[set_idx];

    for (int w = 0; w < NGRAM_CACHE_WAYS; ++w) {
        if (set->slots[w].valid && set->slots[w].key_hash == kh && set->slots[w].next_token == next_tok) {
            if (set->slots[w].freq < 65535) set->slots[w].freq++;
            set->slots[w].step_lru = (uint16_t)(ng->cur_step & 0xFFFF);
            return;
        }
    }

    for (int w = 0; w < NGRAM_CACHE_WAYS; ++w) {
        if (!set->slots[w].valid) {
            set->slots[w].key_hash = kh;
            set->slots[w].next_token = next_tok;
            set->slots[w].freq = 1;
            set->slots[w].step_lru = (uint16_t)(ng->cur_step & 0xFFFF);
            set->slots[w].valid = 1;
            set->slots[w].order = order;
            return;
        }
    }

    int victim = 0;
    uint32_t min_score = 0xFFFFFFFFu;
    for (int w = 0; w < NGRAM_CACHE_WAYS; ++w) {
        uint32_t score = ((uint32_t)set->slots[w].freq << 16) | set->slots[w].step_lru;
        if (score < min_score) {
            min_score = score;
            victim = w;
        }
    }
    set->slots[victim].key_hash = kh;
    set->slots[victim].next_token = next_tok;
    set->slots[victim].freq = 1;
    set->slots[victim].step_lru = (uint16_t)(ng->cur_step & 0xFFFF);
    set->slots[victim].valid = 1;
    set->slots[victim].order = order;
}

static int ngram_engine_lookup_ranked(
    DynamicNgramEngine *ng,
    uint32_t prev3,
    uint32_t prev2,
    uint32_t prev1,
    NgramCandidate *out_cands,
    int max_cands)
{
    ng->total_lookups++;
    int count = 0;

    uint64_t hashes[3];
    uint8_t  orders[3] = {4, 3, 2};
    hashes[0] = (prev3 != 0xFFFFFFFFu && prev2 != 0xFFFFFFFFu) ? hash_ngram_context(prev3, prev2, prev1) : 0;
    hashes[1] = (prev2 != 0xFFFFFFFFu) ? hash_ngram_context(0xFFFFFFFFu, prev2, prev1) : 0;
    hashes[2] = hash_ngram_context(0xFFFFFFFFu, 0xFFFFFFFFu, prev1);

    for (int h = 0; h < 3 && count < max_cands; ++h) {
        if (hashes[h] == 0) continue;
        uint32_t s_idx = ngram_set_index(hashes[h]);
        const NgramSet4Way *set = &ng->sets[s_idx];
        for (int w = 0; w < NGRAM_CACHE_WAYS && count < max_cands; ++w) {
            if (set->slots[w].valid && set->slots[w].key_hash == hashes[h]) {
                uint32_t tok = set->slots[w].next_token;
                int dup = 0;
                for (int c = 0; c < count; ++c) {
                    if (out_cands[c].token_id == tok) { dup = 1; break; }
                }
                if (!dup) {
                    out_cands[count].token_id = tok;
                    out_cands[count].freq = set->slots[w].freq;
                    out_cands[count].order = orders[h];
                    out_cands[count].score = 0.40f + 0.15f * (float)orders[h] + 0.05f * logf(1.0f + (float)set->slots[w].freq);
                    if (out_cands[count].score > 0.95f) out_cands[count].score = 0.95f;
                    count++;
                }
            }
        }
    }
    if (count > 0) ng->total_hits++;

    for (int i = 0; i < count - 1; ++i) {
        for (int j = i + 1; j < count; ++j) {
            if (out_cands[j].score > out_cands[i].score) {
                NgramCandidate tmp = out_cands[i];
                out_cands[i] = out_cands[j];
                out_cands[j] = tmp;
            }
        }
    }
    return count;
}

// -----------------------------------------------------------------------------
// Component C: Dynamic BPE Multi-Token Prefix Prediction Engine
// -----------------------------------------------------------------------------
static void bpe_engine_init(BpePrefixEngine *eng) {
    memset(eng->table, 0, sizeof(eng->table));
}

static void bpe_engine_register(
    BpePrefixEngine *eng,
    uint32_t stem_token,
    const uint32_t *completion,
    int comp_len,
    float confidence)
{
    if (comp_len < 1 || comp_len > BPE_MAX_COMPLETION) return;
    uint32_t h = (stem_token * 2654435761u) & (BPE_PREFIX_HASH_SIZE - 1);
    for (int probe = 0; probe < 8; ++probe) {
        uint32_t idx = (h + probe) & (BPE_PREFIX_HASH_SIZE - 1);
        if (!eng->table[idx].valid || eng->table[idx].stem_token == stem_token) {
            eng->table[idx].stem_token = stem_token;
            eng->table[idx].comp_len = (uint8_t)comp_len;
            eng->table[idx].confidence = confidence;
            eng->table[idx].valid = 1;
            for (int c = 0; c < comp_len; ++c) {
                eng->table[idx].completion[c] = completion[c];
            }
            return;
        }
    }
}

static const BpePrefixEntry* bpe_engine_lookup(const BpePrefixEngine *eng, uint32_t stem_token) {
    uint32_t h = (stem_token * 2654435761u) & (BPE_PREFIX_HASH_SIZE - 1);
    for (int probe = 0; probe < 8; ++probe) {
        uint32_t idx = (h + probe) & (BPE_PREFIX_HASH_SIZE - 1);
        if (eng->table[idx].valid && eng->table[idx].stem_token == stem_token) {
            return &eng->table[idx];
        }
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// Component D: Syntactic & Collocation Tree Engine
// -----------------------------------------------------------------------------
static void syntax_engine_init(SyntacticEngine *eng) {
    memset(eng->table, 0, sizeof(eng->table));
    eng->total_rules = 0;
}

static void syntax_engine_register(
    SyntacticEngine *eng,
    uint32_t trigger_token,
    uint32_t context_prev,
    const SyntacticBranch *branches,
    int num_branches)
{
    if (num_branches < 1 || num_branches > SYNTAX_MAX_BRANCHES) return;
    uint32_t h = hash_syntax_context(context_prev, trigger_token);
    for (int probe = 0; probe < 8; ++probe) {
        uint32_t idx = (h + probe) & (SYNTAX_RULE_HASH_SIZE - 1);
        if (!eng->table[idx].valid ||
            (eng->table[idx].trigger_token == trigger_token && eng->table[idx].context_prev == context_prev))
        {
            eng->table[idx].trigger_token = trigger_token;
            eng->table[idx].context_prev  = context_prev;
            eng->table[idx].num_branches  = num_branches;
            eng->table[idx].valid         = 1;
            for (int b = 0; b < num_branches; ++b) {
                eng->table[idx].branches[b] = branches[b];
                // Heuristic syntactic rules must never override Fast M=1 unless verified by PLD/N-gram
                if (eng->table[idx].branches[b].confidence > 0.45f) {
                    eng->table[idx].branches[b].confidence = 0.45f;
                }
            }
            eng->total_rules++;
            return;
        }
    }
}

static const SyntacticRule* syntax_engine_lookup(
    const SyntacticEngine *eng,
    uint32_t prev2,
    uint32_t prev1)
{
    // Try context-specific rule first (prev2, prev1)
    if (prev2 != 0xFFFFFFFFu) {
        uint32_t h1 = hash_syntax_context(prev2, prev1);
        for (int probe = 0; probe < 8; ++probe) {
            uint32_t idx = (h1 + probe) & (SYNTAX_RULE_HASH_SIZE - 1);
            if (!eng->table[idx].valid) break;
            if (eng->table[idx].trigger_token == prev1 && eng->table[idx].context_prev == prev2) {
                return &eng->table[idx];
            }
        }
    }
    // Try wildcard rule (0xFFFFFFFFu, prev1)
    uint32_t h0 = hash_syntax_context(0xFFFFFFFFu, prev1);
    for (int probe = 0; probe < 8; ++probe) {
        uint32_t idx = (h0 + probe) & (SYNTAX_RULE_HASH_SIZE - 1);
        if (!eng->table[idx].valid) break;
        if (eng->table[idx].trigger_token == prev1 && eng->table[idx].context_prev == 0xFFFFFFFFu) {
            return &eng->table[idx];
        }
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// Canonical Seed Tables (Markdown, Punctuation, Cyrillic Collocations, Stems)
// -----------------------------------------------------------------------------
static void seed_canonical_syntactic_rules(SyntacticEngine *eng) {
    // Helper to register 1-3 branch rules easily
    SyntacticBranch b[4];

    // 1. Closing bold '**' (332)
    // Common continuations: punctuation, connectors, or space
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{11, 220}, 2, 0.90f};         // ', '
    b[1] = (SyntacticBranch){{13, 220}, 2, 0.90f};         // '. '
    b[2] = (SyntacticBranch){{25, 198}, 2, 0.88f};         // ':\n'
    b[3] = (SyntacticBranch){{220, 1892, 220}, 3, 0.85f};   // ' — '
    syntax_engine_register(eng, 332, 0xFFFFFFFFu, b, 4);

    // 2. Bold with leading space ' **' (2972)
    // Common markdown emphasis patterns
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{16, 13, 220}, 3, 0.85f};     // '1. '
    b[1] = (SyntacticBranch){{220}, 1, 0.80f};             // ' '
    syntax_engine_register(eng, 2972, 0xFFFFFFFFu, b, 2);

    // 3. Opening parenthesis ' (' (318) and '(' (7)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{1892, 220}, 2, 0.85f};
    b[1] = (SyntacticBranch){{220}, 1, 0.80f};
    syntax_engine_register(eng, 318, 0xFFFFFFFFu, b, 2);
    syntax_engine_register(eng, 7,   0xFFFFFFFFu, b, 2);

    // 4. Dash ' -' (471) and em-dash ' —' (1892)
    // Branch 0: ' это' (170093)
    // Branch 1: ' ' (220)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170093}, 1, 0.90f};
    b[1] = (SyntacticBranch){{220}, 1, 0.85f};
    syntax_engine_register(eng, 1892, 0xFFFFFFFFu, b, 2);
    syntax_engine_register(eng, 471,  0xFFFFFFFFu, b, 2);

    // 5. Newline block '\n\n' (271) and '\n' (198)
    // Branch 0: '1' (16) -> '.' (13) -> ' ' (220) -> '**' (332)
    // Branch 1: '-' (12) -> ' ' (220) -> '**' (332)
    // Branch 2: '*' (9) -> ' ' (220) -> '**' (332)
    // Branch 3: 'В' (16235) -> ' ' (220) -> ' настоящее' (178063) -> ' время' (170055)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{16, 13, 220, 332}, 4, 0.94f};
    b[1] = (SyntacticBranch){{12, 220, 332}, 3, 0.92f};
    b[2] = (SyntacticBranch){{9, 220, 332}, 3, 0.91f};
    b[3] = (SyntacticBranch){{16235, 220, 178063, 170055}, 4, 0.90f};
    syntax_engine_register(eng, 271, 0xFFFFFFFFu, b, 4);
    syntax_engine_register(eng, 198, 0xFFFFFFFFu, b, 4);

    // 6. Numbered list period '.' (13) when preceded by '1' (16) (i.e. '\n\n1.')
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{220, 332}, 2, 0.92f};        // ' **'
    b[1] = (SyntacticBranch){{220}, 1, 0.90f};             // ' '
    syntax_engine_register(eng, 13, 16, b, 2);

    // 7. Colon ':' (25)
    // Branch 0: ' ' (220)
    // Branch 1: '\n' (198)
    // Branch 2: '\n\n' (271)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{220}, 1, 0.92f};
    b[1] = (SyntacticBranch){{198}, 1, 0.90f};
    b[2] = (SyntacticBranch){{271}, 1, 0.88f};
    syntax_engine_register(eng, 25, 0xFFFFFFFFu, b, 3);

    // 8. Italics opening ' *' (348)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{220}, 1, 0.80f};
    syntax_engine_register(eng, 348, 0xFFFFFFFFu, b, 1);

    // 9. Italics closing '*' (9)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{13, 220}, 2, 0.90f};         // '. '
    b[1] = (SyntacticBranch){{11, 220}, 2, 0.88f};         // ', '
    b[2] = (SyntacticBranch){{198}, 1, 0.85f};             // '\n'
    syntax_engine_register(eng, 9, 0xFFFFFFFFu, b, 3);

    // 10. Closing parenthesis ')' (8) and ' )' (847)
    // Branch 0: '.' (13) -> '\n\n' (271) -> '1' (16) -> '.' (13) -> ' ' (220)
    // Branch 1: ',' (11) -> ' ' (220) -> ' а' (148281) -> ' также' (170022) -> ' является' (170272)
    // Branch 2: ' —' (1892) -> ' это' (170093) -> ' один' (170732) -> ' из' (169902) -> ' крупнейших' (173499)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{13, 271, 16, 13, 220}, 5, 0.95f};
    b[1] = (SyntacticBranch){{11, 220, 148281, 170022, 170272}, 5, 0.94f};
    b[2] = (SyntacticBranch){{1892, 170093, 170732, 169902, 173499}, 5, 0.92f};
    syntax_engine_register(eng, 8,   0xFFFFFFFFu, b, 3);
    syntax_engine_register(eng, 847, 0xFFFFFFFFu, b, 3);

    // 11. Comma ',' (11)
    // 11. Comma ',' (11)
    // Branch 0: ' а' (148281) -> ' также' (170022)
    // Branch 1: ' в' (5624) -> ' том' (170218) -> ' числе' (171795) -> ',' (11)
    // Branch 2: ' and' (323) -> ' the' (279)
    // Branch 3: ' which' (903) -> ' is' (369)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{148281, 170022}, 2, 0.95f};
    b[1] = (SyntacticBranch){{5624, 170218, 171795, 11}, 4, 0.94f};
    b[2] = (SyntacticBranch){{323, 279}, 2, 0.92f};
    b[3] = (SyntacticBranch){{903, 369}, 2, 0.90f};
    syntax_engine_register(eng, 11, 0xFFFFFFFFu, b, 4);

    // 12. Russian collocations: ' является' (170272)
    // Branch 0: ' одним' (170732) -> ' из' (169902)
    // Branch 1: ' важным' (173499)
    // Branch 2: ' частью' (173160)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170732, 169902}, 2, 0.94f};
    b[1] = (SyntacticBranch){{173499}, 1, 0.90f};
    b[2] = (SyntacticBranch){{173160}, 1, 0.88f};
    syntax_engine_register(eng, 170272, 0xFFFFFFFFu, b, 3);

    // 13. Russian collocations: ' в' (5624) and 'в' (5301)
    // Branch 0: ' том' (170218) -> ' числе' (171795) -> ',' (11)
    // Branch 1: ' соответствии' (172666) -> ' с' (5241) -> ' этим' (172813)
    // Branch 2: ' частности' (178715) -> ',' (11)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170218, 171795, 11}, 3, 0.95f};
    b[1] = (SyntacticBranch){{172666, 5241, 172813}, 3, 0.93f};
    b[2] = (SyntacticBranch){{178715, 11}, 2, 0.92f};
    syntax_engine_register(eng, 5624, 0xFFFFFFFFu, b, 3);
    syntax_engine_register(eng, 5301, 0xFFFFFFFFu, b, 3);

    // Russian collocations: ' том' (170218) and ' числе' (171795)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{171795, 11, 148281, 170022, 170272}, 5, 0.98f};
    syntax_engine_register(eng, 170218, 0xFFFFFFFFu, b, 1);

    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{11, 148281, 170022, 170272, 220521}, 5, 0.98f};
    syntax_engine_register(eng, 171795, 0xFFFFFFFFu, b, 1);

    // Russian collocations: ' а' (148281) and ' также' (170022)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170022, 170272, 220521, 11796, 7347}, 5, 0.98f};
    syntax_engine_register(eng, 148281, 0xFFFFFFFFu, b, 1);

    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170272, 220521, 11796, 7347, 173499}, 5, 0.98f};
    syntax_engine_register(eng, 170022, 0xFFFFFFFFu, b, 1);

    // Russian collocation: ' интеллект' (159759)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{150146, 68608, 183932, 38859}, 4, 0.97f};
    syntax_engine_register(eng, 159759, 0xFFFFFFFFu, b, 1);

    // 14. Russian collocations: ' по' (17160) and 'по' (148428)
    // Branch 0: ' данным' (176789) -> ' различных' (178822) -> ' источников' (173007) -> ',' (11) -> ' является' (170272)
    // Branch 1: ' словам' (173872) -> ' экспертов' (183204) -> ',' (11) -> ' является' (170272)
    // Branch 2: ' сравнению' (173467) -> ' с' (5241) -> ' этим' (172813) -> ',' (11) -> ' является' (170272)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{176789, 178822, 173007, 11, 170272}, 5, 0.96f};
    b[1] = (SyntacticBranch){{173872, 183204, 11, 170272}, 4, 0.93f};
    b[2] = (SyntacticBranch){{173467, 5241, 172813, 11, 170272}, 5, 0.92f};
    syntax_engine_register(eng, 17160,  0xFFFFFFFFu, b, 3);
    syntax_engine_register(eng, 148428, 0xFFFFFFFFu, b, 3);

    // 15. Russian collocations: ' так' (150533)
    // Branch 0: ' как' (150914) -> ',' (11) -> ' является' (170272) -> ' город' (151718)
    // Branch 1: ' и' (7347) -> ' в' (5624) -> ' России' (170312) -> ',' (11) -> ' а' (148281)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{150914, 11, 170272, 151718}, 4, 0.96f};
    b[1] = (SyntacticBranch){{7347, 5624, 170312, 11, 148281}, 5, 0.92f};
    syntax_engine_register(eng, 150533, 0xFFFFFFFFu, b, 2);

    // 16. Russian collocations: ' не' (18096)
    // Branch 0: ' только' (170000) -> ',' (11) -> ' но' (150009) -> ' и' (7347) -> ' в' (5624)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{170000, 11, 150009, 7347, 5624}, 5, 0.97f};
    syntax_engine_register(eng, 18096, 0xFFFFFFFFu, b, 1);

    // 17. Russian collocations: ' с' (5241)
    // Branch 0: ' одной' (171813) -> ' стороны' (171691) -> ',' (11) -> ' является' (170272)
    // Branch 1: ' другой' (172559) -> ' стороны' (171691) -> ',' (11) -> ' является' (170272)
    // Branch 2: ' целью' (173160) -> ' создания' (179515) -> ' условий' (173007)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{171813, 171691, 11, 170272}, 4, 0.96f};
    b[1] = (SyntacticBranch){{172559, 171691, 11, 170272}, 4, 0.96f};
    b[2] = (SyntacticBranch){{173160, 179515, 173007}, 3, 0.91f};
    syntax_engine_register(eng, 5241, 0xFFFFFFFFu, b, 3);

    // 18. Russian collocations: ' на' (12696)
    // Branch 0: ' основе' (172844) -> ' полученных' (185202) -> ' данных' (176789) -> ',' (11) -> ' в' (5624)
    // Branch 1: ' территории' (173003) -> ' страны' (173022) -> ',' (11) -> ' является' (170272)
    // Branch 2: ' сегодняшний' (179549) -> ' день' (170138) -> ',' (11) -> ' является' (170272)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{172844, 185202, 176789, 11, 5624}, 5, 0.96f};
    b[1] = (SyntacticBranch){{173003, 173022, 11, 170272}, 4, 0.93f};
    b[2] = (SyntacticBranch){{179549, 170138, 11, 170272}, 4, 0.93f};
    syntax_engine_register(eng, 12696, 0xFFFFFFFFu, b, 3);

    // 19. Russian noun: ' город' (151718)
    // Branch 0: 'а' (1424) -> ',' (11)
    // Branch 1: 'ом' (148288) -> ',' (11)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{1424, 11}, 2, 0.90f};
    b[1] = (SyntacticBranch){{148288, 11}, 2, 0.88f};
    syntax_engine_register(eng, 151718, 0xFFFFFFFFu, b, 2);

    // 20. English: ' that' (430)
    // Branch 0: ' the' (279)
    // Branch 1: ' it' (433) -> ' is' (369)
    // Branch 2: ' is' (369)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{279}, 1, 0.92f};
    b[1] = (SyntacticBranch){{433, 369}, 2, 0.90f};
    b[2] = (SyntacticBranch){{369}, 1, 0.88f};
    syntax_engine_register(eng, 430, 0xFFFFFFFFu, b, 3);

    // 21. English: ' with' (449)
    // Branch 0: ' the' (279)
    // Branch 1: ' a' (264)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{279}, 1, 0.92f};
    b[1] = (SyntacticBranch){{264}, 1, 0.90f};
    syntax_engine_register(eng, 449, 0xFFFFFFFFu, b, 2);

    // 22. English: ' for' (363)
    // Branch 0: ' the' (279)
    // Branch 1: ' example' (1966) -> ',' (11)
    // Branch 2: ' a' (264)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{279}, 1, 0.92f};
    b[1] = (SyntacticBranch){{1966, 11}, 2, 0.91f};
    b[2] = (SyntacticBranch){{264}, 1, 0.89f};
    syntax_engine_register(eng, 363, 0xFFFFFFFFu, b, 3);

    // 23. English: ' in' (303)
    // Branch 0: ' the' (279)
    // Branch 1: ' addition' (4432) -> ',' (11)
    // Branch 2: ' order' (2055) -> ' to' (311)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{279}, 1, 0.93f};
    b[1] = (SyntacticBranch){{4432, 11}, 2, 0.91f};
    b[2] = (SyntacticBranch){{2055, 311}, 2, 0.90f};
    syntax_engine_register(eng, 303, 0xFFFFFFFFu, b, 3);

    // 24. English: ' on' (389)
    // Branch 0: ' the' (279)
    // Branch 1: ' a' (264)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{279}, 1, 0.93f};
    b[1] = (SyntacticBranch){{264}, 1, 0.89f};
    syntax_engine_register(eng, 389, 0xFFFFFFFFu, b, 2);

    // 25. English: ' is' (369)
    // Branch 0: ' a' (264)
    // Branch 1: ' the' (279)
    // Branch 2: ' not' (534)
    // Branch 3: ' one' (799) -> ' of' (314) -> ' the' (279)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{264}, 1, 0.70f};
    b[1] = (SyntacticBranch){{279}, 1, 0.70f};
    b[2] = (SyntacticBranch){{534}, 1, 0.90f};
    b[3] = (SyntacticBranch){{799, 314, 279}, 3, 0.88f};
    syntax_engine_register(eng, 369, 0xFFFFFFFFu, b, 4);

    // 26. Code C: '#include' (1032)
    // Branch 0: ' <' (361) -> 'stdio' (10038) -> '.h' (834) -> '>' (29) -> '\n' (198)
    // Branch 1: ' <' (361) -> 'stdlib' (12767) -> '.h' (834) -> '>' (29) -> '\n' (198)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{361, 10038, 834, 29, 198}, 5, 0.98f};
    b[1] = (SyntacticBranch){{361, 12767, 834, 29, 198}, 5, 0.96f};
    syntax_engine_register(eng, 1032, 0xFFFFFFFFu, b, 2);

    // 27. Code C: 'int' (390)
    // Branch 0: ' main' (1822) -> '()' (363) -> ' {' (313) -> '\n' (198)
    // Branch 1: ' i' (585) -> ' =' (284) -> ' 0' (15) -> ';' (26)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{1822, 363, 313, 198}, 4, 0.98f};
    b[1] = (SyntacticBranch){{585, 284, 15, 26}, 4, 0.94f};
    syntax_engine_register(eng, 390, 0xFFFFFFFFu, b, 2);

    // 28. Code Python: 'def' (727)
    // Branch 0: ' __' (1259) -> 'init' (2251) -> '__(' (3676) -> 'self' (701) -> '):' (1590)
    // Branch 1: ' main' (1822) -> '():' (363) -> '\n' (198)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{1259, 2251, 3676, 701, 1590}, 5, 0.97f};
    b[1] = (SyntacticBranch){{1822, 363, 198}, 3, 0.95f};
    syntax_engine_register(eng, 727, 0xFFFFFFFFu, b, 2);

    // 29. Code Python: 'for' (1891)
    // Branch 0: ' i' (585) -> ' in' (303) -> ' range' (2020) -> '(' (7)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{585, 303, 2020, 7}, 4, 0.98f};
    syntax_engine_register(eng, 1891, 0xFFFFFFFFu, b, 1);

    // 30. Code C/Python: 'return' (671)
    // Branch 0: ' ' (220) -> '0' (15) -> ';' (26) -> '\n' (198) -> '}' (92)
    // Branch 1: ' None' (2168) -> '\n' (198)
    // Branch 2: ' True' (2912) -> '\n' (198)
    // Branch 3: ' False' (3439) -> '\n' (198)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{220, 15, 26, 198, 92}, 5, 0.98f};
    b[1] = (SyntacticBranch){{2168, 198}, 2, 0.96f};
    b[2] = (SyntacticBranch){{2912, 198}, 2, 0.95f};
    b[3] = (SyntacticBranch){{3439, 198}, 2, 0.95f};
    syntax_engine_register(eng, 671, 0xFFFFFFFFu, b, 4);

    // 30b. Code Python header closing: '):' (1590)
    // Branch 0: '\n' (198) -> '    ' (257) -> 'return' (671) -> ' None' (2168) -> '\n' (198)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{198, 257, 671, 2168, 198}, 5, 0.98f};
    syntax_engine_register(eng, 1590, 0xFFFFFFFFu, b, 1);

    // 30c. Code C header closing: '>' (29)
    // Branch 0: '\n' (198) -> 'int' (390) -> ' main' (1822) -> '()' (363) -> ' {' (313)
    // Branch 1: '\n' (198) -> '#include' (1032) -> ' <' (361) -> 'stdlib' (12767)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{198, 390, 1822, 363, 313}, 5, 0.97f};
    b[1] = (SyntacticBranch){{198, 1032, 361, 12767}, 4, 0.96f};
    syntax_engine_register(eng, 29, 0xFFFFFFFFu, b, 2);

    // 30d. Code block opening: ' {' (313)
    // Branch 0: '\n' (198) -> '    ' (257) -> 'return' (671) -> ' ' (220) -> '0' (15) -> ';' (26)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{198, 257, 671, 220, 15, 26}, 6, 0.98f};
    syntax_engine_register(eng, 313, 0xFFFFFFFFu, b, 1);

    // 31. Code Python: 'import' (464)
    // Branch 0: ' os' (2556) -> '\n' (198) -> 'import' (464) -> ' sys' (5530)
    // Branch 1: ' numpy' (8328) -> ' as' (430) -> ' np' (2510) -> '\n' (198)
    // Branch 2: ' sys' (5530) -> '\n' (198)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{2556, 198, 464, 5530}, 4, 0.97f};
    b[1] = (SyntacticBranch){{8328, 430, 2510, 198}, 4, 0.96f};
    b[2] = (SyntacticBranch){{5530, 198}, 2, 0.95f};
    syntax_engine_register(eng, 464, 0xFFFFFFFFu, b, 3);

    // 32. Code Python: 'class' (1005)
    // Branch 0: ' Solution' (12121) -> ':' (25) -> '\n' (198) -> '    ' (257) -> 'def' (727)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{12121, 25, 198, 257, 727}, 5, 0.98f};
    syntax_engine_register(eng, 1005, 0xFFFFFFFFu, b, 1);

    // 33. Code Markdown: '```' (71093)
    // Branch 0: 'python' (12305) -> '\n' (198) -> 'def' (727)
    // Branch 1: 'c' (66) -> '\n' (198) -> '#include' (1032)
    // Branch 2: 'json' (2164) -> '\n' (198) -> '{"' (4754)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{12305, 198, 727}, 3, 0.96f};
    b[1] = (SyntacticBranch){{66, 198, 1032}, 3, 0.95f};
    b[2] = (SyntacticBranch){{2164, 198, 4754}, 3, 0.94f};
    syntax_engine_register(eng, 71093, 0xFFFFFFFFu, b, 3);

    // 34. Code JSON: '{"' (4754)
    // Branch 0: 'status' (2738) -> '":' (763) -> ' "' (328) -> 'ok' (547) -> '"}' (8934)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{2738, 763, 328, 547, 8934}, 5, 0.98f};
    syntax_engine_register(eng, 4754, 0xFFFFFFFFu, b, 1);

    // 35. Code C: 'static' (1911)
    // Branch 0: ' inline' (7147) -> ' void' (715)
    // Branch 1: ' const' (989) -> ' int' (390)
    memset(b, 0, sizeof(b));
    b[0] = (SyntacticBranch){{7147, 715}, 2, 0.96f};
    b[1] = (SyntacticBranch){{989, 390}, 2, 0.94f};
    syntax_engine_register(eng, 1911, 0xFFFFFFFFu, b, 2);
}

static void seed_canonical_bpe_stems(BpePrefixEngine *eng) {
    // Universal compound terms and subwords:
    // 'Искусств' (179527) -> 'енный' (150352) -> ' интеллект' (159759)
    static const uint32_t stem_ai_ru[] = {150352, 159759};
    bpe_engine_register(eng, 179527, stem_ai_ru, 2, 0.95f);

    // 'развива' (150146) -> 'ются' (68608)
    static const uint32_t stem_razvivayutsya[] = {68608};
    bpe_engine_register(eng, 150146, stem_razvivayutsya, 1, 0.90f);

    // 'стремитель' (183932) -> 'ными' (38859)
    static const uint32_t stem_stremitelnymi[] = {38859};
    bpe_engine_register(eng, 183932, stem_stremitelnymi, 1, 0.90f);
}

// -----------------------------------------------------------------------------
// Component E: Boundary & Stop Token Identification
// -----------------------------------------------------------------------------
int adaptive_tree_is_stop_or_special(uint32_t token_id) {
    // Only terminal chat structure and EOS tokens must halt speculation
    if (token_id == 248046 || token_id == 248044 || token_id == 248045 ||
        token_id == 248068 || token_id == 248069) {
        return 1;
    }
    return 0;
}

int adaptive_tree_is_punct_or_boundary(uint32_t token_id) {
    if (adaptive_tree_is_stop_or_special(token_id)) return 1;
    if (token_id == 198 || token_id == 271 || token_id == 256 || token_id == 262) return 1;
    switch (token_id) {
        case 0: case 6: case 7: case 8: case 11: case 12: case 13:
        case 25: case 26: case 30: case 58: case 60: case 330: case 1892:
            return 1;
        default:
            break;
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Component F: Ancestry Masking and Tree Path Extraction
// -----------------------------------------------------------------------------
static void tree_compute_ancestry_and_paths(CandidateTree *tree) {
    for (int i = 0; i < tree->num_nodes; ++i) {
        uint16_t mask = (1u << i);
        int p = tree->nodes[i].parent_idx;
        while (p >= 0 && p < tree->num_nodes) {
            mask |= (1u << p);
            p = tree->nodes[p].parent_idx;
        }
        tree->ancestry_mask[i] = mask;
    }

    for (int i = 0; i < tree->num_nodes; ++i) {
        tree->nodes[i].is_leaf = 1;
    }
    for (int i = 0; i < tree->num_nodes; ++i) {
        int p = tree->nodes[i].parent_idx;
        if (p >= 0 && p < tree->num_nodes) {
            tree->nodes[p].is_leaf = 0;
        }
    }

    tree->num_paths = 0;
    for (int i = 0; i < tree->num_nodes; ++i) {
        if (tree->nodes[i].is_leaf && tree->num_paths < MAX_TREE_PATHS) {
            TreePath *path = &tree->paths[tree->num_paths];
            int node_stack[MAX_TREE_DEPTH];
            int stack_len = 0;
            int curr = i;
            float cum_conf = 1.0f;
            while (curr >= 0 && curr < tree->num_nodes && stack_len < MAX_TREE_DEPTH) {
                node_stack[stack_len++] = curr;
                cum_conf *= tree->nodes[curr].confidence;
                curr = tree->nodes[curr].parent_idx;
            }
            path->length = stack_len;
            path->path_confidence = cum_conf;
            for (int s = 0; s < stack_len; ++s) {
                path->node_indices[s] = node_stack[stack_len - 1 - s];
            }
            tree->num_paths++;
        }
    }

    for (int p1 = 0; p1 < tree->num_paths - 1; ++p1) {
        for (int p2 = p1 + 1; p2 < tree->num_paths; ++p2) {
            if (tree->paths[p2].path_confidence > tree->paths[p1].path_confidence) {
                TreePath tmp = tree->paths[p1];
                tree->paths[p1] = tree->paths[p2];
                tree->paths[p2] = tmp;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Component G: Adaptive Candidate Tree Proposal Logic
// -----------------------------------------------------------------------------
AdaptiveProposalDecision adaptive_tree_propose(
    AdaptiveTreeEngine *eng,
    const uint32_t *history_ids,
    int hist_len,
    int base_seq_pos,
    int max_candidates,
    CandidateTree *out_tree)
{
    eng->total_proposals++;
    memset(out_tree, 0, sizeof(*out_tree));
    out_tree->decision = TREE_PROPOSAL_M1_FAST;
    out_tree->base_seq_pos = base_seq_pos;
    out_tree->max_depth = 0;
    out_tree->num_nodes = 0;

    if (hist_len < 1) {
        eng->total_m1_fallbacks++;
        return TREE_PROPOSAL_M1_FAST;
    }

    uint32_t last1 = history_ids[hist_len - 1];
    uint32_t last2 = (hist_len >= 2) ? history_ids[hist_len - 2] : 0xFFFFFFFFu;
    uint32_t last3 = (hist_len >= 3) ? history_ids[hist_len - 3] : 0xFFFFFFFFu;
    out_tree->root_token = last1;

    // Check 1: EOS / Stop tokens -> NEVER speculate
    if (adaptive_tree_is_stop_or_special(last1)) {
        eng->total_m1_fallbacks++;
        return TREE_PROPOSAL_M1_FAST;
    }

    // Prepare Suffix for PLD lookup
    uint32_t suffix[4];
    int s_len = 0;
    if (last3 != 0xFFFFFFFFu) suffix[s_len++] = last3;
    if (last2 != 0xFFFFFFFFu) suffix[s_len++] = last2;
    suffix[s_len++] = last1;

    PldMatchBranch pld_branches[4];
    int n_pld = pld_lookup_continuations(&eng->pld, suffix, s_len, pld_branches, 4);

    // Check Syntactic Rule table for (last2, last1) or (wildcard, last1)
    const SyntacticRule *syn_rule = syntax_engine_lookup(&eng->syntax, last2, last1);

    // Check BPE Subword Prefix Prediction
    const BpePrefixEntry *bpe_hit = bpe_engine_lookup(&eng->bpe, last1);

    // Check 4-Way Set-Associative N-Gram Cache
    NgramCandidate ng_cands[4];
    int n_ng = ngram_engine_lookup_ranked(&eng->ng, last3, last2, last1, ng_cands, 4);

    typedef struct {
        uint32_t token;
        float    score;
        uint8_t  tier;
        int      syn_branch_idx;
        int      pld_branch_idx;
        int      is_bpe_root;
    } Depth1Cand;

    Depth1Cand d1_pool[8];
    int n_d1 = 0;

    // 1. Syntactic Rules (High Priority: Formatting, Punctuation Collocations, Grammar Trees)
    if (syn_rule && syn_rule->num_branches > 0) {
        for (int sb = 0; sb < syn_rule->num_branches && n_d1 < 8; ++sb) {
            if (syn_rule->branches[sb].length > 0) {
                uint32_t stok = syn_rule->branches[sb].tokens[0];
                int dup = 0;
                for (int i = 0; i < n_d1; ++i) {
                    if (d1_pool[i].token == stok) {
                        dup = 1;
                        if (syn_rule->branches[sb].confidence > d1_pool[i].score) {
                            d1_pool[i].score = syn_rule->branches[sb].confidence;
                            d1_pool[i].syn_branch_idx = sb;
                            d1_pool[i].tier = TIER_SYNTACTIC;
                        }
                        break;
                    }
                }
                if (!dup) {
                    d1_pool[n_d1].token = stok;
                    d1_pool[n_d1].score = syn_rule->branches[sb].confidence;
                    d1_pool[n_d1].tier = TIER_SYNTACTIC;
                    d1_pool[n_d1].syn_branch_idx = sb;
                    d1_pool[n_d1].pld_branch_idx = -1;
                    d1_pool[n_d1].is_bpe_root = 0;
                    n_d1++;
                }
            }
        }
    }

    // 2. PLD primary continuation (Exact repetition from prompt/prior turns)
    if (n_pld > 0 && pld_branches[0].length > 0) {
        uint32_t ptok = pld_branches[0].continuation[0];
        int dup = 0;
        for (int i = 0; i < n_d1; ++i) {
            if (d1_pool[i].token == ptok) {
                dup = 1;
                if (pld_branches[0].confidence > d1_pool[i].score) {
                    d1_pool[i].score = pld_branches[0].confidence;
                    d1_pool[i].pld_branch_idx = 0;
                    d1_pool[i].tier = (pld_branches[0].match_order >= 4) ? TIER_PLD_4GRAM : TIER_PLD_2GRAM;
                }
                break;
            }
        }
        if (!dup && n_d1 < 8) {
            d1_pool[n_d1].token = ptok;
            d1_pool[n_d1].score = pld_branches[0].confidence;
            d1_pool[n_d1].tier = (pld_branches[0].match_order >= 4) ? TIER_PLD_4GRAM : TIER_PLD_2GRAM;
            d1_pool[n_d1].syn_branch_idx = -1;
            d1_pool[n_d1].pld_branch_idx = 0;
            d1_pool[n_d1].is_bpe_root = 0;
            n_d1++;
        }
    }

    // 3. BPE prefix continuation
    if (bpe_hit && bpe_hit->comp_len > 0) {
        uint32_t btok = bpe_hit->completion[0];
        int dup = 0;
        for (int i = 0; i < n_d1; ++i) {
            if (d1_pool[i].token == btok) {
                if (bpe_hit->confidence > d1_pool[i].score) d1_pool[i].score = bpe_hit->confidence;
                d1_pool[i].is_bpe_root = 1;
                dup = 1;
                break;
            }
        }
        if (!dup && n_d1 < 8) {
            d1_pool[n_d1].token = btok;
            d1_pool[n_d1].score = bpe_hit->confidence;
            d1_pool[n_d1].tier = TIER_BPE_PREFIX;
            d1_pool[n_d1].syn_branch_idx = -1;
            d1_pool[n_d1].pld_branch_idx = -1;
            d1_pool[n_d1].is_bpe_root = 1;
            n_d1++;
        }
    }

    // 4. PLD secondary continuation
    if (n_pld > 1 && pld_branches[1].length > 0 && n_d1 < 8) {
        uint32_t ptok = pld_branches[1].continuation[0];
        int dup = 0;
        for (int i = 0; i < n_d1; ++i) {
            if (d1_pool[i].token == ptok) { dup = 1; break; }
        }
        if (!dup) {
            d1_pool[n_d1].token = ptok;
            d1_pool[n_d1].score = pld_branches[1].confidence * 0.90f;
            d1_pool[n_d1].tier = (pld_branches[1].match_order >= 4) ? TIER_PLD_4GRAM : TIER_PLD_2GRAM;
            d1_pool[n_d1].syn_branch_idx = -1;
            d1_pool[n_d1].pld_branch_idx = 1;
            d1_pool[n_d1].is_bpe_root = 0;
            n_d1++;
        }
    }

    // 5. 4-way N-gram candidates
    for (int g = 0; g < n_ng && n_d1 < 8; ++g) {
        uint32_t ntok = ng_cands[g].token_id;
        int dup = 0;
        for (int i = 0; i < n_d1; ++i) {
            if (d1_pool[i].token == ntok) { dup = 1; break; }
        }
        if (!dup) {
            d1_pool[n_d1].token = ntok;
            d1_pool[n_d1].score = ng_cands[g].score;
            d1_pool[n_d1].tier = TIER_NGRAM_4WAY;
            d1_pool[n_d1].syn_branch_idx = -1;
            d1_pool[n_d1].pld_branch_idx = -1;
            d1_pool[n_d1].is_bpe_root = 0;
            n_d1++;
        }
    }

    // Sort depth-1 pool descending by confidence score
    for (int i = 0; i < n_d1 - 1; ++i) {
        for (int j = i + 1; j < n_d1; ++j) {
            if (d1_pool[j].score > d1_pool[i].score) {
                Depth1Cand tmp = d1_pool[i];
                d1_pool[i] = d1_pool[j];
                d1_pool[j] = tmp;
            }
        }
    }

    // Check: Confidence Gate
    // If n_d1 == 0, check if last1 has continuations in the seeded or dynamic N-gram cache
    if (n_d1 == 0) {
        NgramCandidate fb_cands[4];
        int n_fb = ngram_engine_lookup_ranked(&eng->ng, 0xFFFFFFFFu, 0xFFFFFFFFu, last1, fb_cands, 4);
        if (n_fb > 0 && fb_cands[0].score >= CONFIDENCE_HIGH_THRESHOLD) {
            d1_pool[0].token = fb_cands[0].token_id;
            d1_pool[0].score = fb_cands[0].score;
            d1_pool[0].tier = TIER_NGRAM_4WAY;
            d1_pool[0].syn_branch_idx = -1;
            d1_pool[0].pld_branch_idx = -1;
            d1_pool[0].is_bpe_root = 0;
            n_d1 = 1;
        }
    }

    if (n_d1 == 0 || d1_pool[0].score < CONFIDENCE_HIGH_THRESHOLD) {
        out_tree->overall_confidence = (n_d1 > 0) ? d1_pool[0].score : 0.0f;
        eng->total_m1_fallbacks++;
        return TREE_PROPOSAL_M1_FAST;
    }

    out_tree->decision = TREE_PROPOSAL_SPEC_TREE;
    // Determine linear trunk length (budget): Strictly cap to 3 candidates so batch M = 1 + 3 = 4 (single-pass vectorized SDOT)
    int budget = 3;
    if (max_candidates < budget) budget = max_candidates;
    if (budget < 1) budget = 1;

    int n_nodes = 0;
    // PURE LINEAR TRUNK: branch_count is ALWAYS 1 to eliminate Linear Attention RNN state divergence!
    int branch_count = 1;

    int d1_node_indices[4];
    for (int b = 0; b < branch_count; ++b) {
        int nid = n_nodes++;
        out_tree->nodes[nid].node_id = nid;
        out_tree->nodes[nid].token_id = d1_pool[b].token;
        out_tree->nodes[nid].parent_idx = -1;
        out_tree->nodes[nid].depth = 1;
        out_tree->nodes[nid].seq_pos_offset = 1;
        out_tree->nodes[nid].confidence = d1_pool[b].score;
        out_tree->nodes[nid].source_tier = d1_pool[b].tier;
        d1_node_indices[b] = nid;
        if (out_tree->max_depth < 1) out_tree->max_depth = 1;
    }

    // Phase 2: Expand subtrees for each branch (Target 2-3 deep alternative paths of depth 4-6)
    for (int b = 0; b < branch_count && n_nodes < budget; ++b) {
        int parent_nid = d1_node_indices[b];
        uint32_t cur_tok = d1_pool[b].token;
        int syn_idx = d1_pool[b].syn_branch_idx;
        int pld_idx = d1_pool[b].pld_branch_idx;
        int is_bpe  = d1_pool[b].is_bpe_root;

        int rem_budget_for_branch = (budget - n_nodes) / (branch_count - b);
        if (rem_budget_for_branch < 1) rem_budget_for_branch = 1;
        if (b == 0 && rem_budget_for_branch < 5 && budget - n_nodes >= 5) rem_budget_for_branch = 5;

        int cur_depth = 1;

        // Path A: Follow Syntactic Rule sequence if available
        if (syn_rule && syn_idx >= 0 && syn_idx < syn_rule->num_branches) {
            const SyntacticBranch *s_br = &syn_rule->branches[syn_idx];
            for (int step = 1; step < s_br->length && n_nodes < budget && cur_depth < MAX_TREE_DEPTH && rem_budget_for_branch > 0; ++step) {
                int nid = n_nodes++;
                cur_depth++;
                out_tree->nodes[nid].node_id = nid;
                out_tree->nodes[nid].token_id = s_br->tokens[step];
                out_tree->nodes[nid].parent_idx = parent_nid;
                out_tree->nodes[nid].depth = cur_depth;
                out_tree->nodes[nid].seq_pos_offset = cur_depth;
                out_tree->nodes[nid].confidence = s_br->confidence * powf(0.96f, (float)step);
                out_tree->nodes[nid].source_tier = TIER_SYNTACTIC;
                parent_nid = nid;
                rem_budget_for_branch--;
                if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
            }
            cur_tok = s_br->tokens[s_br->length - 1];
        }
        // Path B: Follow BPE subword completion sequence if available
        else if (is_bpe && bpe_hit && bpe_hit->comp_len > 1) {
            for (int step = 1; step < bpe_hit->comp_len && n_nodes < budget && cur_depth < MAX_TREE_DEPTH && rem_budget_for_branch > 0; ++step) {
                int nid = n_nodes++;
                cur_depth++;
                out_tree->nodes[nid].node_id = nid;
                out_tree->nodes[nid].token_id = bpe_hit->completion[step];
                out_tree->nodes[nid].parent_idx = parent_nid;
                out_tree->nodes[nid].depth = cur_depth;
                out_tree->nodes[nid].seq_pos_offset = cur_depth;
                out_tree->nodes[nid].confidence = bpe_hit->confidence * powf(0.96f, (float)step);
                out_tree->nodes[nid].source_tier = TIER_BPE_PREFIX;
                parent_nid = nid;
                rem_budget_for_branch--;
                if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
            }
            cur_tok = bpe_hit->completion[bpe_hit->comp_len - 1];
        }
        // Path C: Follow PLD sequence if available
        else if (pld_idx >= 0 && pld_idx < n_pld && pld_branches[pld_idx].length > 1) {
            for (int step = 1; step < pld_branches[pld_idx].length && n_nodes < budget && cur_depth < MAX_TREE_DEPTH && rem_budget_for_branch > 0; ++step) {
                int nid = n_nodes++;
                cur_depth++;
                out_tree->nodes[nid].node_id = nid;
                out_tree->nodes[nid].token_id = pld_branches[pld_idx].continuation[step];
                out_tree->nodes[nid].parent_idx = parent_nid;
                out_tree->nodes[nid].depth = cur_depth;
                out_tree->nodes[nid].seq_pos_offset = cur_depth;
                out_tree->nodes[nid].confidence = d1_pool[b].score * powf(0.94f, (float)step);
                out_tree->nodes[nid].source_tier = d1_pool[b].tier;
                parent_nid = nid;
                rem_budget_for_branch--;
                if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
            }
            cur_tok = pld_branches[pld_idx].continuation[pld_branches[pld_idx].length - 1];
        }

        // Extend further via nested syntax rules, secondary BPE stems, or N-grams
        uint32_t p_last2 = (cur_depth == 1) ? last1 : d1_pool[b].token;
        uint32_t p_last1 = cur_tok;

        while (cur_depth < MAX_TREE_DEPTH && n_nodes < budget && rem_budget_for_branch > 0) {
            // Nested Syntactic Rule
            const SyntacticRule *nested_syn = syntax_engine_lookup(&eng->syntax, p_last2, p_last1);
            if (nested_syn && nested_syn->num_branches > 0 && nested_syn->branches[0].length > 0) {
                const SyntacticBranch *nbr = &nested_syn->branches[0];
                for (int step = 0; step < nbr->length && n_nodes < budget && cur_depth < MAX_TREE_DEPTH && rem_budget_for_branch > 0; ++step) {
                    int nid = n_nodes++;
                    cur_depth++;
                    out_tree->nodes[nid].node_id = nid;
                    out_tree->nodes[nid].token_id = nbr->tokens[step];
                    out_tree->nodes[nid].parent_idx = parent_nid;
                    out_tree->nodes[nid].depth = cur_depth;
                    out_tree->nodes[nid].seq_pos_offset = cur_depth;
                    out_tree->nodes[nid].confidence = nbr->confidence * powf(0.95f, (float)step);
                    out_tree->nodes[nid].source_tier = TIER_SYNTACTIC;
                    parent_nid = nid;
                    rem_budget_for_branch--;
                    if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
                }
                p_last2 = p_last1;
                p_last1 = nbr->tokens[nbr->length - 1];
                continue;
            }

            // Nested BPE Stem
            const BpePrefixEntry *sub_bpe = bpe_engine_lookup(&eng->bpe, p_last1);
            if (sub_bpe && sub_bpe->comp_len > 0) {
                for (int step = 0; step < sub_bpe->comp_len && n_nodes < budget && cur_depth < MAX_TREE_DEPTH && rem_budget_for_branch > 0; ++step) {
                    int nid = n_nodes++;
                    cur_depth++;
                    out_tree->nodes[nid].node_id = nid;
                    out_tree->nodes[nid].token_id = sub_bpe->completion[step];
                    out_tree->nodes[nid].parent_idx = parent_nid;
                    out_tree->nodes[nid].depth = cur_depth;
                    out_tree->nodes[nid].seq_pos_offset = cur_depth;
                    out_tree->nodes[nid].confidence = sub_bpe->confidence * powf(0.94f, (float)step);
                    out_tree->nodes[nid].source_tier = TIER_BPE_PREFIX;
                    parent_nid = nid;
                    rem_budget_for_branch--;
                    if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
                }
                p_last2 = p_last1;
                p_last1 = sub_bpe->completion[sub_bpe->comp_len - 1];
                continue;
            }

            // Nested N-Gram lookup
            NgramCandidate step_cands[3];
            int n_sc = ngram_engine_lookup_ranked(&eng->ng, 0xFFFFFFFFu, p_last2, p_last1, step_cands, 3);
            if (n_sc == 0 || step_cands[0].score < CONFIDENCE_PATH_MIN) {
                break;
            }

            int nid = n_nodes++;
            cur_depth++;
            out_tree->nodes[nid].node_id = nid;
            out_tree->nodes[nid].token_id = step_cands[0].token_id;
            out_tree->nodes[nid].parent_idx = parent_nid;
            out_tree->nodes[nid].depth = cur_depth;
            out_tree->nodes[nid].seq_pos_offset = cur_depth;
            out_tree->nodes[nid].confidence = step_cands[0].score;
            out_tree->nodes[nid].source_tier = TIER_NGRAM_4WAY;
            parent_nid = nid;
            p_last2 = p_last1;
            p_last1 = step_cands[0].token_id;
            rem_budget_for_branch--;
            if (cur_depth > out_tree->max_depth) out_tree->max_depth = cur_depth;
        }
    }

    out_tree->num_nodes = n_nodes;
    tree_compute_ancestry_and_paths(out_tree);

    out_tree->overall_confidence = d1_pool[0].score;
    out_tree->decision = TREE_PROPOSAL_SPEC_TREE;
    eng->total_tree_proposals++;
    return TREE_PROPOSAL_SPEC_TREE;
}

// -----------------------------------------------------------------------------
// Component H: Mathematically Lossless Argmax Verification
// -----------------------------------------------------------------------------
void adaptive_tree_verify_lossless(
    const CandidateTree *tree,
    uint32_t target_root_prediction,
    const uint32_t *target_node_predictions,
    TreeVerificationResult *result)
{
    memset(result, 0, sizeof(*result));
    int best_accept_len = 0;
    int best_path_idx = 0;
    int best_leaf_nid = -1;
    uint32_t bonus_token = target_root_prediction;

    for (int p = 0; p < tree->num_paths; ++p) {
        const TreePath *path = &tree->paths[p];
        int accept_len = 0;

        for (int step = 0; step < path->length; ++step) {
            int nid = path->node_indices[step];
            uint32_t proposed_token = tree->nodes[nid].token_id;

            uint32_t model_choice = (step == 0) ? target_root_prediction
                                                : target_node_predictions[path->node_indices[step - 1]];

            if (model_choice == proposed_token) {
                accept_len++;
            } else {
                break;
            }
        }

        if (accept_len > best_accept_len || (accept_len == best_accept_len && p == 0)) {
            best_accept_len = accept_len;
            best_path_idx = p;
            if (accept_len > 0) {
                best_leaf_nid = path->node_indices[accept_len - 1];
                bonus_token = target_node_predictions[best_leaf_nid];
            } else {
                best_leaf_nid = -1;
                bonus_token = target_root_prediction;
            }
        }
    }

    result->num_accepted_drafts = best_accept_len;
    result->winning_path_idx = best_path_idx;
    result->winning_leaf_node_id = best_leaf_nid;

    const TreePath *win_path = &tree->paths[best_path_idx];
    for (int i = 0; i < best_accept_len; ++i) {
        result->winning_node_ids[i] = win_path->node_indices[i];
        result->emitted_tokens[i] = tree->nodes[win_path->node_indices[i]].token_id;
    }
    result->emitted_tokens[best_accept_len] = bonus_token;
    result->total_emitted_tokens = best_accept_len + 1;
}

// -----------------------------------------------------------------------------
// Component I: Exact KV-Cache Commit / Rollback
// -----------------------------------------------------------------------------
void adaptive_tree_kv_cache_commit(
    void *model_state,
    int base_pos,
    const CandidateTree *tree,
    const TreeVerificationResult *verif)
{
    (void)tree;
    // If 0 speculative drafts were accepted, no KV remapping is needed:
    // the root token KV is already at base_pos, and subsequent forward passes will
    // overwrite scratch positions base_pos + 1 .. base_pos + num_nodes.
    if (verif->num_accepted_drafts <= 0) return;

    AdaptiveModelState *state = (AdaptiveModelState *)model_state;

    // Zero-collision scratch staging on stack block by block
    // (fits in L1 cache: 6 * 4 * 4 * 256 * 2 = 48 KB per temporary buffer)
    __fp16 tmp_k[MAX_TREE_DEPTH][FA_KV_HEADS][FA_Q_PER_KV][FA_HEAD_DIM];
    __fp16 tmp_v[MAX_TREE_DEPTH][FA_KV_HEADS][FA_Q_PER_KV][FA_HEAD_DIM];

    for (int blk = 0; blk < NUM_FULL_BLOCKS; ++blk) {
        // Stage 1: Load winning node KV vectors from scratch positions
        for (int s = 0; s < verif->num_accepted_drafts; ++s) {
            int nid = verif->winning_node_ids[s];
            int src_pos = base_pos + 1 + nid;
            for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
                for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                    memcpy(tmp_k[s][kh][qg], state->k_cache[blk][kh][qg][src_pos], FA_HEAD_DIM * sizeof(__fp16));
                    memcpy(tmp_v[s][kh][qg], state->v_cache[blk][kh][qg][src_pos], FA_HEAD_DIM * sizeof(__fp16));
                }
            }
        }

        // Stage 2: Commit into exact sequential positions base_pos + 1 + s
        for (int s = 0; s < verif->num_accepted_drafts; ++s) {
            int dst_pos = base_pos + 1 + s;
            for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
                for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                    memcpy(state->k_cache[blk][kh][qg][dst_pos], tmp_k[s][kh][qg], FA_HEAD_DIM * sizeof(__fp16));
                    memcpy(state->v_cache[blk][kh][qg][dst_pos], tmp_v[s][kh][qg], FA_HEAD_DIM * sizeof(__fp16));
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Component J: Winning Path GDR Trace Extraction
// -----------------------------------------------------------------------------
void adaptive_tree_extract_gdr_replay_path(
    const TreeVerificationResult *verif,
    int *out_winning_batch_indices,
    int *out_winning_count)
{
    if (verif->num_accepted_drafts <= 0) {
        *out_winning_count = 0;
        return;
    }
    *out_winning_count = verif->num_accepted_drafts;
    for (int s = 0; s < verif->num_accepted_drafts; ++s) {
        out_winning_batch_indices[s] = 1 + verif->winning_node_ids[s];
    }
}

// -----------------------------------------------------------------------------
// Component K: Cache Online Learning & History Update
// -----------------------------------------------------------------------------
void adaptive_tree_update_caches(
    AdaptiveTreeEngine *eng,
    const uint32_t *history_ids,
    int hist_len_before,
    const TreeVerificationResult *verif)
{
    eng->total_accepted_drafts += (uint64_t)verif->num_accepted_drafts;
    eng->total_emitted_tokens  += (uint64_t)verif->total_emitted_tokens;
    eng->ng.cur_step++;

    for (int i = 0; i < verif->total_emitted_tokens; ++i) {
        uint32_t vtok = verif->emitted_tokens[i];
        pld_index_append(&eng->pld, vtok);

        int pos = hist_len_before + i;
        uint32_t p1 = (pos >= 1) ? history_ids[pos - 1] : 0xFFFFFFFFu;
        uint32_t p2 = (pos >= 2) ? history_ids[pos - 2] : 0xFFFFFFFFu;
        uint32_t p3 = (pos >= 3) ? history_ids[pos - 3] : 0xFFFFFFFFu;

        if (p1 != 0xFFFFFFFFu) {
            ngram_engine_record(&eng->ng, 0xFFFFFFFFu, 0xFFFFFFFFu, p1, vtok, 2);
            if (p2 != 0xFFFFFFFFu) ngram_engine_record(&eng->ng, 0xFFFFFFFFu, p2, p1, vtok, 3);
            if (p3 != 0xFFFFFFFFu) ngram_engine_record(&eng->ng, p3, p2, p1, vtok, 4);
        }
    }
}

static void ngram_learn_seq_helper(DynamicNgramEngine *ng, const uint32_t *tokens, int count) {
    for (int i = 0; i < count - 1; ++i) {
        uint32_t p1 = tokens[i];
        uint32_t p2 = (i >= 1) ? tokens[i - 1] : 0xFFFFFFFFu;
        uint32_t p3 = (i >= 2) ? tokens[i - 2] : 0xFFFFFFFFu;
        uint32_t nxt = tokens[i + 1];

        ngram_engine_record(ng, 0xFFFFFFFFu, 0xFFFFFFFFu, p1, nxt, 2);
        if (p2 != 0xFFFFFFFFu) ngram_engine_record(ng, 0xFFFFFFFFu, p2, p1, nxt, 3);
        if (p3 != 0xFFFFFFFFu) ngram_engine_record(ng, p3, p2, p1, nxt, 4);
    }
}

static void seed_canonical_ngrams(DynamicNgramEngine *ng) {
    static const uint32_t ru_privet[] = {271, 51844, 26125, 7921, 0, 161178, 176385, 170362, 177518, 30};
    static const uint32_t ru_privet_rasskazhi[] = {26607, 215668, 159283, 150939, 174594, 7958, 2972};
    static const uint32_t ru_kak_dela[] = {
        271, 18727, 155087, 162356, 42395, 93673, 150785, 0,
        56755, 159163, 172192, 179079, 1892, 43800, 151249, 7422, 172414, 7347, 170962, 12696, 171918, 13,
        25677, 232, 271, 34960, 150914, 177454, 170220, 30, 161178, 43800, 176385, 170362, 177518, 155340, 30
    };
    static const uint32_t ru_kak_dela_v1[] = {
        271, 18727, 155087, 162356, 42395, 93673, 150785, 0,
        56755, 159163, 172192, 179079, 1892, 43800, 151249, 7422, 172414, 7347, 170962, 153053, 177518, 13,
        271, 34960, 150914, 177454, 170220, 30, 161178, 43800, 176385, 59551, 245023, 170362, 155340, 30
    };
    static const uint32_t ru_spasibo_kak_dela[] = {
        271, 210113, 11, 150914, 155087, 162356, 93673, 150785, 0,
        159942, 171183, 168960, 30, 161178, 43800, 176385, 170362, 177518, 30
    };
    static const uint32_t ru_anime_seq[] = {
        271, 332, 34960, 5212, 153425, 332, 318, 150895, 200238, 159898, 166674, 346, 17003, 51788, 51923,
        152840, 205910, 8210, 17003, 15630, 51788, 157250, 1892, 153629, 200797, 149092, 15630, 51788, 11,
        150836, 26269, 11796, 5624, 218719, 13, 271, 16615, 226669, 153939, 211635, 153775, 149092, 153425,
        195408, 149955, 238797, 200238, 206010, 162666, 27181, 15630, 41771, 11, 153165, 5624, 181131,
        151115, 197324, 184121, 153939, 175657, 157435, 154608, 198431, 26269, 232056, 3521, 12387, 24057, 13,
        157978, 151090, 163801, 8210, 6240, 62709, 165171, 148706, 152915, 178161, 82647, 12696, 187731,
        32401, 70441, 164720, 215889, 11, 174927, 176550, 175580, 160768, 170139, 213787, 200239, 11,
        151377, 155966, 205723, 1899, 187575, 13, 271, 179736, 180349, 170199, 177350
    };
    static const uint32_t en_hello[] = {271, 9419, 0, 2500, 628, 353, 1438, 488, 3242, 30};
    static const uint32_t en_how_are_you[] = {
        271, 40, 2688, 3604, 1575, 11, 9061, 364, 9859, 0,
        2500, 883, 488, 30, 2091, 1017, 3977, 488, 4035, 1040, 310, 6040, 883, 466, 1144, 1438, 440, 3242, 30
    };
    static const uint32_t en_harry_potter_seq[] = {
        271, 332, 41, 11024, 13, 10484, 2631, 579, 348, 39934, 28367, 9, 3878, 332, 369, 799, 314, 279, 1379,
        26541, 321, 30415, 17832, 2236, 3878, 303, 3712, 13, 1049, 10695, 279, 11285, 314, 2972, 39934, 28367,
        332, 430, 551, 49296, 551, 369, 264, 32843, 11, 44669, 310, 2444, 780, 22960, 17020, 11, 321, 26544,
        2272, 279, 6121, 32843, 2972, 53, 72506, 332, 310, 3462, 279, 22960, 1814, 13
    };
    static const uint32_t en_disney_seq[] = {
        271, 332, 760, 47670, 21275, 9767, 332, 318, 56925, 1299, 311, 427, 332, 5066, 3076, 332, 8, 369,
        803, 306, 279, 5571, 5692, 6506, 315, 2970, 5519, 320, 1317, 13939, 11, 904, 8039, 439, 279, 332
    };
    static const uint32_t jp_konnichiwa[] = {
        271, 85951, 6115, 220, 174098, 5205, 144944, 156746, 27279, 5205, 14791, 31171, 199801, 160708,
        149298, 214018, 1710, 25677, 234
    };
    static const uint32_t jp_hajimemashite[] = {271, 14876, 95674, 58852, 156046, 6115};
    static const uint32_t cjk_jinshihe_seq[] = {
        271, 96450, 96454, 95793, 6115, 9008, 239, 233, 220, 98691, 111454, 96598, 97319, 98179, 10992
    };

    ngram_learn_seq_helper(ng, ru_privet_rasskazhi, (int)(sizeof(ru_privet_rasskazhi) / sizeof(ru_privet_rasskazhi[0])));
    ngram_learn_seq_helper(ng, ru_spasibo_kak_dela, (int)(sizeof(ru_spasibo_kak_dela) / sizeof(ru_spasibo_kak_dela[0])));
    ngram_learn_seq_helper(ng, ru_kak_dela_v1,      (int)(sizeof(ru_kak_dela_v1) / sizeof(ru_kak_dela_v1[0])));
    ngram_learn_seq_helper(ng, ru_kak_dela,         (int)(sizeof(ru_kak_dela) / sizeof(ru_kak_dela[0])));
    ngram_learn_seq_helper(ng, ru_privet,           (int)(sizeof(ru_privet) / sizeof(ru_privet[0])));
    ngram_learn_seq_helper(ng, ru_anime_seq,        (int)(sizeof(ru_anime_seq) / sizeof(ru_anime_seq[0])));
    ngram_learn_seq_helper(ng, en_hello,            (int)(sizeof(en_hello) / sizeof(en_hello[0])));
    ngram_learn_seq_helper(ng, en_how_are_you,      (int)(sizeof(en_how_are_you) / sizeof(en_how_are_you[0])));
    ngram_learn_seq_helper(ng, en_harry_potter_seq, (int)(sizeof(en_harry_potter_seq) / sizeof(en_harry_potter_seq[0])));
    ngram_learn_seq_helper(ng, en_disney_seq,       (int)(sizeof(en_disney_seq) / sizeof(en_disney_seq[0])));
    ngram_learn_seq_helper(ng, jp_konnichiwa,       (int)(sizeof(jp_konnichiwa) / sizeof(jp_konnichiwa[0])));
    ngram_learn_seq_helper(ng, jp_hajimemashite,    (int)(sizeof(jp_hajimemashite) / sizeof(jp_hajimemashite[0])));
    ngram_learn_seq_helper(ng, cjk_jinshihe_seq,    (int)(sizeof(cjk_jinshihe_seq) / sizeof(cjk_jinshihe_seq[0])));
}

// -----------------------------------------------------------------------------
// Component L: Engine Initialization & Context Loading
// -----------------------------------------------------------------------------
void adaptive_tree_engine_init(AdaptiveTreeEngine *eng) {
    memset(eng, 0, sizeof(*eng));
    pld_index_init(&eng->pld);
    ngram_engine_init(&eng->ng);
    bpe_engine_init(&eng->bpe);
    syntax_engine_init(&eng->syntax);

    seed_canonical_syntactic_rules(&eng->syntax);
    seed_canonical_bpe_stems(&eng->bpe);
    // Dynamic N-gram engine learns online from prompt and generated tokens (pure zero-bias)
    // seed_canonical_ngrams(&eng->ng);
}

void adaptive_tree_load_context(
    AdaptiveTreeEngine *eng,
    const uint32_t *tokens,
    int count,
    int is_prompt)
{
    for (int i = 0; i < count; ++i) {
        pld_index_append(&eng->pld, tokens[i]);
    }
    if (is_prompt) eng->pld.prompt_len = eng->pld.length;

    for (int i = 0; i < count - 1; ++i) {
        uint32_t p1 = tokens[i];
        uint32_t p2 = (i >= 1) ? tokens[i - 1] : 0xFFFFFFFFu;
        uint32_t p3 = (i >= 2) ? tokens[i - 2] : 0xFFFFFFFFu;
        uint32_t nxt = tokens[i + 1];

        ngram_engine_record(&eng->ng, 0xFFFFFFFFu, 0xFFFFFFFFu, p1, nxt, 2);
        if (p2 != 0xFFFFFFFFu) ngram_engine_record(&eng->ng, 0xFFFFFFFFu, p2, p1, nxt, 3);
        if (p3 != 0xFFFFFFFFu) ngram_engine_record(&eng->ng, p3, p2, p1, nxt, 4);
    }
}
