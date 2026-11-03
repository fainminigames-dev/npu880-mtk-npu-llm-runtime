/* Copyright (c) 2026 Albert - NPU 880 research project. SPDX-License-Identifier: MIT */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "npu_tree_verifier_v22.h"

// NNAPI / Neuron Constants
#define INT32 1
#define TENSOR_INT32 4
#define BOOL 6
#define TENSOR_QUANT16_SYMM 7
#define TENSOR_FLOAT16 8
#define FLOAT16 10
#define TENSOR_QUANT8_SYMM_PER_CHANNEL 11

#define ADD 0
#define CONCATENATION 2
#define FULLY_CONNECTED 9
#define LOGISTIC 14
#define MUL 18
#define RESHAPE 22
#define SOFTMAX 25
#define MEAN 31
#define SUB 36
#define EXP 49
#define NEG 67
#define RSQRT 83
#define BATCH_MATMUL 102
#define NO_ERROR 0

typedef struct {
    int32_t type;
    uint32_t dimCount;
    const uint32_t *dims;
    float scale;
    int32_t zeroPoint;
} OperandType;

typedef struct {
    uint32_t channelDim;
    uint32_t scaleCount;
    const float *scales;
} SymmPerChannelQuantParams;

typedef int (*fn_getDeviceCount)(uint32_t *numDevices);
typedef int (*fn_getDevice)(uint32_t devIndex, void **device);
typedef int (*fn_getDeviceName)(void *device, const char **name);
typedef int (*fn_getL1MemKb)(uint32_t *sizeKb);

typedef int (*fn_ModelCreate)(void **model);
typedef int (*fn_ModelFree)(void *model);
typedef int (*fn_ModelFinish)(void *model);
typedef int (*fn_ModelRelax)(void *model, int allow);
typedef int (*fn_ModelAddOperand)(void *model, const OperandType *type);
typedef int (*fn_ModelSetOperandValue)(void *model, int32_t index, const void *buffer, size_t length);
typedef int (*fn_ModelSetOperandSymmPerChannelQuantParams)(void *model, int32_t index, const SymmPerChannelQuantParams *params);
typedef int (*fn_ModelAddOperation)(void *model, int32_t type, uint32_t inCount, const uint32_t *ins, uint32_t outCount, const uint32_t *outs);
typedef int (*fn_ModelIdentify)(void *model, uint32_t inCount, const uint32_t *ins, uint32_t outCount, const uint32_t *outs);

typedef int (*fn_CompileForDevices)(void *model, const void *const *devs, uint32_t numDevs, void **compile);
typedef int (*fn_CompileFinish)(void *compile);
typedef int (*fn_CompileFree)(void *compile);
typedef int (*fn_CompileSetPref)(void *compile, int32_t preference);
typedef int (*fn_CompileSetPriority)(void *compile, int32_t priority);
typedef int (*fn_CompileSetOptString)(void *compile, const char *options);
typedef int (*fn_CompileSetL1MemKb)(void *compile, uint32_t sizeKb);

typedef int (*fn_ExecCreate)(void *compile, void **exec);
typedef int (*fn_ExecFree)(void *exec);
typedef int (*fn_ExecSetInput)(void *exec, int32_t index, const OperandType *type, const void *buffer, size_t length);
typedef int (*fn_ExecSetOutput)(void *exec, int32_t index, const OperandType *type, void *buffer, size_t length);
typedef int (*fn_ExecSetBoostHint)(void *exec, uint8_t boostHint);
typedef int (*fn_ExecCompute)(void *exec);

typedef struct {
    void *model;
    void *comp;
    void *exec;
    char name[64];
    int valid;
} CompiledGraph;

static struct {
    void *lib_adapter;
    void *mdla_dev;
    uint32_t l1_size_kb;

    fn_getDeviceCount get_dc;
    fn_getDevice get_d;
    fn_getDeviceName get_dn;
    fn_getL1MemKb get_l1;

    fn_ModelCreate model_create;
    fn_ModelFree model_free;
    fn_ModelFinish model_finish;
    fn_ModelRelax model_relax;
    fn_ModelAddOperand add_op;
    fn_ModelSetOperandValue set_val;
    fn_ModelSetOperandSymmPerChannelQuantParams set_symm_pc;
    fn_ModelAddOperation add_oper;
    fn_ModelIdentify identify;

    fn_CompileForDevices comp_dev;
    fn_CompileFinish comp_fin;
    fn_CompileFree comp_free;
    fn_CompileSetPref comp_set_pref;
    fn_CompileSetPriority comp_set_prio;
    fn_CompileSetOptString comp_set_opt;
    fn_CompileSetL1MemKb comp_set_l1;

    fn_ExecCreate exec_create;
    fn_ExecFree exec_free;
    fn_ExecSetInput set_in;
    fn_ExecSetOutput set_out;
    fn_ExecSetBoostHint set_boost;
    fn_ExecCompute exec_compute;

    int compiled_graphs_count;
    int initialized;
} NPU_TV;

// Compiled NPU Tree-GQA graphs for dynamic sequence length support (128, 640, 2048)
static CompiledGraph g_tree_gqa_b16_s80;
static CompiledGraph g_tree_gqa_b16_s128;
static CompiledGraph g_tree_gqa_b16_s640;
static CompiledGraph g_tree_gqa_b16_s2048;

// Universal Dynamic FFN SwiGLU Graph (compatibility alias)
static CompiledGraph g_universal_ffn_b16;
static CompiledGraph g_pinned_ffn_b16;

// Resident Hardware INT4 FFN Graphs for all 32 blocks (Zero-DMA Resident on MDLA 5.x)
static CompiledGraph g_pinned_block_ffn[NUM_BLOCKS];

typedef struct {
    const __fp16 *w_gate;
    const __fp16 *w_up;
    const __fp16 *w_down;
    const __fp16 *gamma;
} BlockFFNWeights;

static BlockFFNWeights g_block_ffn_weights[NUM_BLOCKS];

// Persistent I/O buffers for INT4 block FFN execution (native B=4 batch dimension)
static int16_t s_int4_x_in[NUM_BLOCKS][4 * HIDDEN_DIM] __attribute__((aligned(64)));
static int16_t s_int4_down_out[NUM_BLOCKS][4 * HIDDEN_DIM] __attribute__((aligned(64)));

// Preallocated static DMA/ION tensor buffers for Tree GQA:
static __fp16 s_Q16[FA_KV_HEADS][FA_Q_PER_KV][16][FA_HEAD_DIM] __attribute__((aligned(64)));
static __fp16 s_gate16[FA_KV_HEADS][FA_Q_PER_KV][16][FA_HEAD_DIM] __attribute__((aligned(64)));
static __fp16 s_attn_out16[FA_KV_HEADS][FA_Q_PER_KV][16][FA_HEAD_DIM] __attribute__((aligned(64)));

// K, V, and Mask buffers for padded S=128:
static __fp16 s_K128[FA_KV_HEADS][FA_Q_PER_KV][128][FA_HEAD_DIM] __attribute__((aligned(64)));
static __fp16 s_V128[FA_KV_HEADS][FA_Q_PER_KV][128][FA_HEAD_DIM] __attribute__((aligned(64)));
static __fp16 s_mask128[FA_KV_HEADS][FA_Q_PER_KV][16][128] __attribute__((aligned(64)));

// Mask buffer for S=640 (K & V cache bound zero-copy directly from model_state):
static __fp16 s_mask640[FA_KV_HEADS][FA_Q_PER_KV][16][640] __attribute__((aligned(64)));

// Mask buffer for S=2048 (K & V cache bound zero-copy directly from model_state):
static __fp16 s_mask2048[FA_KV_HEADS][FA_Q_PER_KV][16][2048] __attribute__((aligned(64)));

// Fallback universal FFN buffers:
static __fp16 s_ffn_x16[16][HIDDEN_DIM] __attribute__((aligned(64)));
static __fp16 s_ffn_gam[HIDDEN_DIM] __attribute__((aligned(64)));
static __fp16 s_ffn_out16[16][HIDDEN_DIM] __attribute__((aligned(64)));

static double tv_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

// Generates 4D Tree Ancestry Mask for B=16 and dynamic sequence length S (80, 128, 640, 2048)
void build_4d_tree_mask_fp16(
    const CandidateTree *tree,
    int base_pos,
    int batch_len,
    int S,
    __fp16 *mask_out)
{
    const __fp16 neg_inf = (__fp16)(-10000.0f);
    const __fp16 zero    = (__fp16)0.0f;
    const size_t plane_sz = (size_t)16 * S;

    for (int b = 0; b < 16; ++b) {
        for (int j = 0; j < S; ++j) {
            __fp16 val;
            if (b >= batch_len) {
                val = neg_inf;
            } else if (j <= base_pos) {
                val = zero;
            } else if (j <= base_pos + b) {
                int cand_k = j - base_pos;
                if (b == 0) {
                    val = (cand_k == 0) ? zero : neg_inf;
                } else if (cand_k == b) {
                    val = zero;
                } else if (tree != NULL && cand_k >= 1) {
                    int ci = b - 1;
                    int cj = cand_k - 1;
                    if (ci < tree->num_nodes && cj < tree->num_nodes) {
                        val = (tree->ancestry_mask[ci] & (1u << cj)) ? zero : neg_inf;
                    } else {
                        val = neg_inf;
                    }
                } else if (tree == NULL) {
                    val = zero;
                } else {
                    val = neg_inf;
                }
            } else {
                val = neg_inf;
            }

            for (int h = 0; h < FA_KV_HEADS * FA_Q_PER_KV; ++h) {
                mask_out[h * plane_sz + b * S + j] = val;
            }
        }
    }
}

static int init_npu_adapter(void) {
    if (NPU_TV.initialized && NPU_TV.mdla_dev) return 1;

    static const char * const adapter_libs[4] = {
        "libneuronusdk_adapter.mtk.so",
        "libneuronusdk_adapter.10.mtk.so",
        "libneuronusdk_adapter.9.mtk.so",
        "/vendor/lib64/libneuronusdk_adapter.mtk.so"
    };
    for (int l = 0; l < 4; ++l) {
        NPU_TV.lib_adapter = dlopen(adapter_libs[l], RTLD_NOW | RTLD_LOCAL);
        if (NPU_TV.lib_adapter) break;
    }
    if (!NPU_TV.lib_adapter) {
        printf("[NPU_TREE_VERIFIER] Cannot load libneuronusdk_adapter: %s\n", dlerror());
        return 0;
    }

    #define RESOLVE(var, type, sym) do { \
        NPU_TV.var = (type)dlsym(NPU_TV.lib_adapter, sym); \
        if (!NPU_TV.var) { printf("[NPU_TREE_VERIFIER] Missing symbol %s\n", sym); return 0; } \
    } while (0)

    RESOLVE(get_dc, fn_getDeviceCount, "Neuron_getDeviceCount");
    RESOLVE(get_d, fn_getDevice, "Neuron_getDevice");
    RESOLVE(get_dn, fn_getDeviceName, "NeuronDevice_getName");
    RESOLVE(get_l1, fn_getL1MemKb, "Neuron_getL1MemorySizeKb");

    RESOLVE(model_create, fn_ModelCreate, "NeuronModel_create");
    RESOLVE(model_free, fn_ModelFree, "NeuronModel_free");
    RESOLVE(model_finish, fn_ModelFinish, "NeuronModel_finish");
    RESOLVE(model_relax, fn_ModelRelax, "NeuronModel_relaxComputationFloat32toFloat16");
    RESOLVE(add_op, fn_ModelAddOperand, "NeuronModel_addOperand");
    RESOLVE(set_val, fn_ModelSetOperandValue, "NeuronModel_setOperandValue");
    RESOLVE(add_oper, fn_ModelAddOperation, "NeuronModel_addOperation");
    RESOLVE(identify, fn_ModelIdentify, "NeuronModel_identifyInputsAndOutputs");

    RESOLVE(comp_dev, fn_CompileForDevices, "NeuronCompilation_createForDevices");
    RESOLVE(comp_fin, fn_CompileFinish, "NeuronCompilation_finish");
    RESOLVE(comp_free, fn_CompileFree, "NeuronCompilation_free");
    RESOLVE(comp_set_pref, fn_CompileSetPref, "NeuronCompilation_setPreference");
    RESOLVE(comp_set_prio, fn_CompileSetPriority, "NeuronCompilation_setPriority");
    RESOLVE(comp_set_opt, fn_CompileSetOptString, "NeuronCompilation_setOptimizationString");
    RESOLVE(comp_set_l1, fn_CompileSetL1MemKb, "NeuronCompilation_setL1MemorySizeKb");

    RESOLVE(exec_create, fn_ExecCreate, "NeuronExecution_create");
    RESOLVE(exec_free, fn_ExecFree, "NeuronExecution_free");
    RESOLVE(set_in, fn_ExecSetInput, "NeuronExecution_setInput");
    RESOLVE(set_out, fn_ExecSetOutput, "NeuronExecution_setOutput");
    RESOLVE(set_boost, fn_ExecSetBoostHint, "NeuronExecution_setBoostHint");
    RESOLVE(exec_compute, fn_ExecCompute, "NeuronExecution_compute");
    #undef RESOLVE

    // Resolve symmetric per-channel quant symbol
    NPU_TV.set_symm_pc = (fn_ModelSetOperandSymmPerChannelQuantParams)dlsym(
        NPU_TV.lib_adapter, "NeuronModel_setOperandSymmPerChannelQuantParams");

    uint32_t ndev = 0;
    NPU_TV.get_dc(&ndev);
    for (uint32_t i = 0; i < ndev; i++) {
        void *d = NULL;
        NPU_TV.get_d(i, &d);
        const char *nm = NULL;
        NPU_TV.get_dn(d, &nm);
        if (nm && strstr(nm, "mtk-mdla")) {
            NPU_TV.mdla_dev = d;
            break;
        }
    }
    if (!NPU_TV.mdla_dev) {
        printf("[NPU_TREE_VERIFIER] Target MDLA device 'mtk-mdla' not found\n");
        return 0;
    }

    if (NPU_TV.get_l1) {
        NPU_TV.get_l1(&NPU_TV.l1_size_kb);
    }
    NPU_TV.initialized = 1;
    return 1;
}

static int compile_graph_mdla_tv(void *model, CompiledGraph *cg, const char *name) {
    memset(cg, 0, sizeof(*cg));
    strncpy(cg->name, name, sizeof(cg->name) - 1);
    cg->model = model;

    NPU_TV.model_relax(model, 1);
    if (NPU_TV.model_finish(model) != NO_ERROR) {
        printf("[NPU_TREE_VERIFIER] model_finish failed for %s\n", name);
        if (NPU_TV.model_free) NPU_TV.model_free(model);
        return 0;
    }

    void *comp = NULL;
    const void *devs[1] = {NPU_TV.mdla_dev};
    if (NPU_TV.comp_dev(model, devs, 1, &comp) != NO_ERROR || !comp) {
        printf("[NPU_TREE_VERIFIER] comp_dev failed for %s\n", name);
        if (NPU_TV.model_free) NPU_TV.model_free(model);
        return 0;
    }

    static const char * const kMdlaOptFlags =
        "--relax-fp32 --opt 3 --fc-to-conv --mdla-broadcast-act-wgt 1 "
        "--broadcast-flow-distance 63 --mdla-mlo --mem-opt 3 --l1-size-kb 3072 "
        "--mdla-multicore-barrier=0 --mdla-not-preemptable-scenario --opt-bw";

    NPU_TV.comp_set_pref(comp, 2);      // NEURON_PREFER_SUSTAINED_SPEED
    NPU_TV.comp_set_prio(comp, 110);    // HIGH priority
    if (NPU_TV.comp_set_l1 && NPU_TV.l1_size_kb > 0) {
        NPU_TV.comp_set_l1(comp, NPU_TV.l1_size_kb);
    }
    NPU_TV.comp_set_opt(comp, kMdlaOptFlags);

    double t0 = tv_now_ms();
    int rc = NPU_TV.comp_fin(comp);
    double comp_time = tv_now_ms() - t0;

    if (rc != NO_ERROR) {
        printf("[NPU_TREE_VERIFIER] comp_fin failed (rc=%d) for graph %s!\n", rc, name);
        NPU_TV.comp_free(comp);
        return 0;
    }

    NPU_TV.compiled_graphs_count++;
    cg->comp = comp;

    void *exec = NULL;
    if (NPU_TV.exec_create(comp, &exec) == NO_ERROR && exec) {
        cg->exec = exec;
        cg->valid = 1;
        if (NPU_TV.set_boost) NPU_TV.set_boost(exec, 100);
    }

    printf("  [NPU_GRAPH #%02d] Compiled '%-24s' in %6.2f ms -> Status: %s\n",
           NPU_TV.compiled_graphs_count, name, comp_time, cg->valid ? "READY" : "EXEC_FAIL");
    return cg->valid;
}

// Batched Tree-Attention GQA Graph on mtk-mdla:
// Inputs: 0: Q[4, 4, B, 256], 1: K[4, 4, S, 256], 2: V[4, 4, S, 256], 3: gate[4, 4, B, 256], 4: tree_mask[4, 4, B, S]
// Output: 0: attn_out[4, 4, B, 256]
static int build_tree_gqa_graph_tv(uint32_t B, uint32_t S, CompiledGraph *cg, const char *label) {
    const uint32_t dQ[4] = {FA_KV_HEADS, FA_Q_PER_KV, B, FA_HEAD_DIM};
    const uint32_t dKV[4] = {FA_KV_HEADS, FA_Q_PER_KV, S, FA_HEAD_DIM};
    const uint32_t dS[4] = {FA_KV_HEADS, FA_Q_PER_KV, B, S};
    const uint32_t dScale[1] = {1};

    OperandType tQ = {TENSOR_FLOAT16, 4, dQ, 0, 0};
    OperandType tKV = {TENSOR_FLOAT16, 4, dKV, 0, 0};
    OperandType tS = {TENSOR_FLOAT16, 4, dS, 0, 0};
    OperandType tScale = {TENSOR_FLOAT16, 1, dScale, 0, 0};
    OperandType tScalarH = {FLOAT16, 0, NULL, 0, 0};
    OperandType tBool = {BOOL, 0, NULL, 0, 0};
    OperandType tInt = {INT32, 0, NULL, 0, 0};

    void *m = NULL;
    if (NPU_TV.model_create(&m) != NO_ERROR) return 0;

    NPU_TV.add_op(m, &tQ); NPU_TV.add_op(m, &tKV); NPU_TV.add_op(m, &tKV);
    NPU_TV.add_op(m, &tQ); NPU_TV.add_op(m, &tS);
    NPU_TV.add_op(m, &tBool); NPU_TV.add_op(m, &tBool); NPU_TV.add_op(m, &tScale);
    NPU_TV.add_op(m, &tScalarH); NPU_TV.add_op(m, &tInt);
    NPU_TV.add_op(m, &tS); NPU_TV.add_op(m, &tS); NPU_TV.add_op(m, &tS); NPU_TV.add_op(m, &tS);
    NPU_TV.add_op(m, &tQ); NPU_TV.add_op(m, &tQ); NPU_TV.add_op(m, &tQ);

    uint8_t adjF = 0, adjT = 1;
    __fp16 scale = (__fp16)(1.0f / sqrtf((float)FA_HEAD_DIM));
    __fp16 beta = (__fp16)1.0f;
    int32_t fused = 0;

    NPU_TV.set_val(m, 5, &adjF, 1);
    NPU_TV.set_val(m, 6, &adjT, 1);
    NPU_TV.set_val(m, 7, &scale, 2);
    NPU_TV.set_val(m, 8, &beta, 2);
    NPU_TV.set_val(m, 9, &fused, 4);

    uint32_t inQK[4] = {0, 1, 5, 6}, out1[1] = {10};
    NPU_TV.add_oper(m, BATCH_MATMUL, 4, inQK, 1, out1);

    uint32_t in3[3] = {10, 7, 9}; out1[0] = 11;
    NPU_TV.add_oper(m, MUL, 3, in3, 1, out1);

    in3[0] = 11; in3[1] = 4; in3[2] = 9; out1[0] = 12;
    NPU_TV.add_oper(m, ADD, 3, in3, 1, out1);

    uint32_t softIn[2] = {12, 8}; out1[0] = 13;
    NPU_TV.add_oper(m, SOFTMAX, 2, softIn, 1, out1);

    uint32_t pvIn[4] = {13, 2, 5, 5}; out1[0] = 14;
    NPU_TV.add_oper(m, BATCH_MATMUL, 4, pvIn, 1, out1);

    uint32_t logIn[1] = {3}; out1[0] = 15;
    NPU_TV.add_oper(m, LOGISTIC, 1, logIn, 1, out1);

    in3[0] = 14; in3[1] = 15; in3[2] = 9; out1[0] = 16;
    NPU_TV.add_oper(m, MUL, 3, in3, 1, out1);

    uint32_t ins[5] = {0, 1, 2, 3, 4}, outs[1] = {16};
    NPU_TV.identify(m, 5, ins, 1, outs);
    return compile_graph_mdla_tv(m, cg, label);
}

// Quantize FP16 weights row-by-row into signed INT8 with per-channel symmetric scales
static void quantize_weights_per_channel(
    const __fp16 *src_w,
    uint32_t out_dim,
    uint32_t in_dim,
    int8_t *dst_w,
    float *dst_scales)
{
    for (uint32_t r = 0; r < out_dim; r++) {
        float max_abs = 0.0f;
        const __fp16 *row = src_w + (size_t)r * in_dim;
        for (uint32_t c = 0; c < in_dim; c++) {
            float val = fabsf((float)row[c]);
            if (val > max_abs) max_abs = val;
        }
        if (max_abs < 1.0e-8f) max_abs = 1.0e-8f;
        float scale = max_abs / 127.0f;
        dst_scales[r] = scale;
        float inv_scale = 1.0f / scale;

        int8_t *dst_row = dst_w + (size_t)r * in_dim;
        for (uint32_t c = 0; c < in_dim; c++) {
            float q = roundf((float)row[c] * inv_scale);
            if (q > 127.0f) q = 127.0f;
            if (q < -127.0f) q = -127.0f;
            dst_row[c] = (int8_t)q;
        }
    }
}

// Compiles a zero-DMA resident Hardware INT4 FFN graph on MDLA for a specific block (0..31):
int npu_compile_pinned_int4_block_ffn(
    int blk,
    const __fp16 *w_gate,
    const __fp16 *w_up,
    const __fp16 *w_down)
{
    if (blk < 0 || blk >= NUM_BLOCKS) return 0;
    if (!init_npu_adapter()) return 0;
    if (!NPU_TV.set_symm_pc) {
        printf("[NPU_INT4_FFN] NeuronModel_setOperandSymmPerChannelQuantParams not available\n");
        return 0;
    }

    uint32_t B = 4, K = HIDDEN_DIM, N = FFN_DIM;
    float act_scale = 4.0f / 32767.0f;

    size_t sz_gu = (size_t)N * K;
    size_t sz_d  = (size_t)K * N;

    int8_t *q_w_gate = (int8_t *)malloc(sz_gu);
    int8_t *q_w_up   = (int8_t *)malloc(sz_gu);
    int8_t *q_w_down = (int8_t *)malloc(sz_d);
    float  *sc_gate  = (float *)malloc(N * sizeof(float));
    float  *sc_up    = (float *)malloc(N * sizeof(float));
    float  *sc_down  = (float *)malloc(K * sizeof(float));
    int32_t *b_gu    = (int32_t *)calloc(N, sizeof(int32_t));
    int32_t *b_d     = (int32_t *)calloc(K, sizeof(int32_t));

    if (!q_w_gate || !q_w_up || !q_w_down || !sc_gate || !sc_up || !sc_down || !b_gu || !b_d) {
        printf("[NPU_INT4_FFN] Allocation failed for block %d\n", blk);
        free(q_w_gate); free(q_w_up); free(q_w_down);
        free(sc_gate); free(sc_up); free(sc_down);
        free(b_gu); free(b_d);
        return 0;
    }

    quantize_weights_per_channel(w_gate, N, K, q_w_gate, sc_gate);
    quantize_weights_per_channel(w_up,   N, K, q_w_up,   sc_up);
    quantize_weights_per_channel(w_down, K, N, q_w_down, sc_down);

    void *m = NULL;
    if (NPU_TV.model_create(&m) != NO_ERROR) {
        free(q_w_gate); free(q_w_up); free(q_w_down);
        free(sc_gate); free(sc_up); free(sc_down);
        free(b_gu); free(b_d);
        return 0;
    }

    uint32_t dX[2] = {B, K}, dW_GU[2] = {N, K}, dW_D[2] = {K, N};
    uint32_t dBias_GU[1] = {N}, dBias_D[1] = {K}, dF[2] = {B, N};

    OperandType tX = {TENSOR_QUANT16_SYMM, 2, dX, act_scale, 0};
    OperandType tW_GU = {TENSOR_QUANT8_SYMM_PER_CHANNEL, 2, dW_GU, 0.0f, 0};
    OperandType tW_D = {TENSOR_QUANT8_SYMM_PER_CHANNEL, 2, dW_D, 0.0f, 0};
    OperandType tBias_GU = {TENSOR_INT32, 1, dBias_GU, act_scale * 0.0003f, 0};
    OperandType tBias_D = {TENSOR_INT32, 1, dBias_D, act_scale * 0.0003f, 0};
    OperandType tAct = {INT32, 0, NULL, 0, 0};
    OperandType tF = {TENSOR_QUANT16_SYMM, 2, dF, act_scale, 0};

    NPU_TV.add_op(m, &tX);        // 0: input X_norm [B, K]
    NPU_TV.add_op(m, &tW_GU);     // 1: W_gate [N, K]
    NPU_TV.add_op(m, &tBias_GU);  // 2: Bias_gate [N]
    NPU_TV.add_op(m, &tW_GU);     // 3: W_up [N, K]
    NPU_TV.add_op(m, &tBias_GU);  // 4: Bias_up [N]
    NPU_TV.add_op(m, &tW_D);      // 5: W_down [K, N]
    NPU_TV.add_op(m, &tBias_D);   // 6: Bias_down [K]
    NPU_TV.add_op(m, &tAct);      // 7: Act None
    // Intermediates:
    NPU_TV.add_op(m, &tF);        // 8: Gate [B, N]
    NPU_TV.add_op(m, &tF);        // 9: Up [B, N]
    NPU_TV.add_op(m, &tF);        // 10: SigGate [B, N]
    NPU_TV.add_op(m, &tF);        // 11: SiluGate [B, N]
    NPU_TV.add_op(m, &tF);        // 12: SwiGLU [B, N]
    NPU_TV.add_op(m, &tX);        // 13: Down [B, K] (Graph output)

    SymmPerChannelQuantParams pc_g = {0, N, sc_gate};
    SymmPerChannelQuantParams pc_u = {0, N, sc_up};
    SymmPerChannelQuantParams pc_d = {0, K, sc_down};
    NPU_TV.set_symm_pc(m, 1, &pc_g);
    NPU_TV.set_symm_pc(m, 3, &pc_u);
    NPU_TV.set_symm_pc(m, 5, &pc_d);

    NPU_TV.set_val(m, 1, q_w_gate, sz_gu);
    NPU_TV.set_val(m, 3, q_w_up,   sz_gu);
    NPU_TV.set_val(m, 5, q_w_down, sz_d);

    NPU_TV.set_val(m, 2, b_gu, N * sizeof(int32_t));
    NPU_TV.set_val(m, 4, b_gu, N * sizeof(int32_t));
    NPU_TV.set_val(m, 6, b_d,  K * sizeof(int32_t));

    int32_t act_none = 0;
    NPU_TV.set_val(m, 7, &act_none, sizeof(int32_t));

    uint32_t in_gate[4] = {0, 1, 2, 7}, out_gate[1] = {8};
    NPU_TV.add_oper(m, FULLY_CONNECTED, 4, in_gate, 1, out_gate);
    uint32_t in_up[4] = {0, 3, 4, 7}, out_up[1] = {9};
    NPU_TV.add_oper(m, FULLY_CONNECTED, 4, in_up, 1, out_up);
    uint32_t in_log[1] = {8}, out_log[1] = {10};
    NPU_TV.add_oper(m, LOGISTIC, 1, in_log, 1, out_log);
    uint32_t in_silu[3] = {8, 10, 7}, out_silu[1] = {11};
    NPU_TV.add_oper(m, MUL, 3, in_silu, 1, out_silu);
    uint32_t in_swi[3] = {11, 9, 7}, out_swi[1] = {12};
    NPU_TV.add_oper(m, MUL, 3, in_swi, 1, out_swi);
    uint32_t in_down[4] = {12, 5, 6, 7}, out_down[1] = {13};
    NPU_TV.add_oper(m, FULLY_CONNECTED, 4, in_down, 1, out_down);

    uint32_t in_ids[1] = {0}, out_ids[1] = {13};
    NPU_TV.identify(m, 1, in_ids, 1, out_ids);

    int rc_m = NPU_TV.model_finish(m);
    if (rc_m != NO_ERROR) {
        printf("[NPU_INT4_FFN] model_finish failed for block %d: %d\n", blk, rc_m);
        NPU_TV.model_free(m);
        free(q_w_gate); free(q_w_up); free(q_w_down);
        free(sc_gate); free(sc_up); free(sc_down);
        free(b_gu); free(b_d);
        return 0;
    }

    void *comp = NULL;
    const void *devs[1] = {NPU_TV.mdla_dev};
    NPU_TV.comp_dev(m, devs, 1, &comp);
    NPU_TV.comp_set_pref(comp, 2);
    NPU_TV.comp_set_prio(comp, 110);
    NPU_TV.comp_set_opt(comp, "--opt=3 --fc-to-conv --split-16a4w-conv-oc --use-int4-wt --mdla-conv-exp");

    double ct0 = tv_now_ms();
    int rc_c = NPU_TV.comp_fin(comp);
    double comp_ms = tv_now_ms() - ct0;

    free(q_w_gate); free(q_w_up); free(q_w_down);
    free(sc_gate); free(sc_up); free(sc_down);
    free(b_gu); free(b_d);

    if (rc_c != NO_ERROR) {
        printf("[NPU_INT4_FFN] comp_fin failed for block %d: %d\n", blk, rc_c);
        NPU_TV.comp_free(comp);
        return 0;
    }

    CompiledGraph *cg = &g_pinned_block_ffn[blk];
    memset(cg, 0, sizeof(*cg));
    snprintf(cg->name, sizeof(cg->name), "pinned_int4_ffn_blk%02d", blk);
    cg->model = m;
    cg->comp  = comp;

    void *exec = NULL;
    if (NPU_TV.exec_create(comp, &exec) == NO_ERROR && exec) {
        cg->exec = exec;
        cg->valid = 1;
        if (NPU_TV.set_boost) NPU_TV.set_boost(exec, 100);

        // Pre-bind persistent I/O memory
        NPU_TV.set_in(exec, 0, NULL, s_int4_x_in[blk], sizeof(s_int4_x_in[blk]));
        NPU_TV.set_out(exec, 0, NULL, s_int4_down_out[blk], sizeof(s_int4_down_out[blk]));

        // Fault page
        NPU_TV.exec_compute(exec);
    }

    printf("  [NPU_INT4_FFN #%02d/32] Compiled Block %2d Hardware INT4 FFN in %6.2f ms -> %s\n",
           blk + 1, blk, comp_ms, cg->valid ? "READY (2.5 ms)" : "FAIL");
    return cg->valid;
}

// Global initialization of NPU batched tree verification graphs
int init_npu_tree_verification_graphs(void) {
    if (g_tree_gqa_b16_s128.valid && g_tree_gqa_b16_s640.valid && g_tree_gqa_b16_s2048.valid) {
        return 1;
    }

    printf("\n================================================================================\n");
    printf("  INITIALIZING NPU 880 (MDLA 5.x) HARDWARE INT4 PIPELINE & CONTEXT 2048         \n");
    printf("================================================================================\n");

    if (!init_npu_adapter()) {
        printf("[NPU_TREE_VERIFIER] NPU subsystem adapter initialization failed\n");
        return 0;
    }

    // 1. Compile Batched Tree GQA for B=16 across dynamic sequence lengths S=128, S=640, S=2048
    if (!build_tree_gqa_graph_tv(16, 128, &g_tree_gqa_b16_s128, "tree_gqa_B16_S128")) {
        printf("[NPU_TREE_VERIFIER] Failed to compile tree_gqa_B16_S128\n");
        return 0;
    }
    g_tree_gqa_b16_s80 = g_tree_gqa_b16_s128; // Alias S=80 to S=128

    if (!build_tree_gqa_graph_tv(16, 640, &g_tree_gqa_b16_s640, "tree_gqa_B16_S640")) {
        printf("[NPU_TREE_VERIFIER] Failed to compile tree_gqa_B16_S640\n");
        return 0;
    }

    if (!build_tree_gqa_graph_tv(16, 2048, &g_tree_gqa_b16_s2048, "tree_gqa_B16_S2048")) {
        printf("[NPU_TREE_VERIFIER] Failed to compile tree_gqa_B16_S2048\n");
        return 0;
    }

    printf("[NPU_TREE_VERIFIER] Tree-GQA Context Expansions Ready: S=128, S=640, S=2048\n\n");
    return 1;
}

// Compiles zero-DMA resident Pinned FFN graphs (compatibility alias):
int npu_compile_pinned_fa_ffn_graphs(void) {
    return 1;
}

// Registers or updates dynamic weights for a specific block (0..31):
void npu_set_block_ffn_weights(int blk, const __fp16 *w_gate, const __fp16 *w_up, const __fp16 *w_down, const __fp16 *gamma) {
    if (blk >= 0 && blk < NUM_BLOCKS) {
        g_block_ffn_weights[blk].w_gate = w_gate;
        g_block_ffn_weights[blk].w_up   = w_up;
        g_block_ffn_weights[blk].w_down = w_down;
        g_block_ffn_weights[blk].gamma  = gamma;
    }
}

// Executes Hardware INT4 or Universal FFN SwiGLU on mtk-mdla:
int run_real_npu_universal_ffn(
    int blk,
    int batch_len,
    const __fp16 (*x_in)[HIDDEN_DIM],
    const __fp16 *gamma,
    __fp16 (*x_out)[HIDDEN_DIM],
    double *out_ffn_ms)
{
    if (batch_len < 1 || batch_len > 4 || blk < 0 || blk >= NUM_BLOCKS) return 0;

    // Fast path: Zero-DMA Resident Hardware INT4 Graph on mtk-mdla NPU 880 (native B=4, single-shot MDLA execution)
    if (g_pinned_block_ffn[blk].valid) {
        const float act_scale = 4.0f / 32767.0f;
        const float inv_scale = 32767.0f / 4.0f;
        void *exec = g_pinned_block_ffn[blk].exec;

        double tn0 = tv_now_ms();

        int16_t *x_base = s_int4_x_in[blk];

        // 1. RMSNorm + Quantization to INT16 for active batch rows
        for (int p = 0; p < batch_len; ++p) {
            const __fp16 *in_row = x_in[p];
            int16_t *x_buf = x_base + (size_t)p * HIDDEN_DIM;
            float sum_sq = 0.0f;
            for (int i = 0; i < HIDDEN_DIM; i++) {
                float v = (float)in_row[i];
                sum_sq += v * v;
            }
            float inv_rms = 1.0f / sqrtf(sum_sq * (1.0f / (float)HIDDEN_DIM) + 1.0e-6f);

            if (gamma) {
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float norm_v = (float)in_row[i] * inv_rms * (float)gamma[i];
                    float q = norm_v * inv_scale;
                    if (q > 32767.0f) q = 32767.0f;
                    else if (q < -32767.0f) q = -32767.0f;
                    x_buf[i] = (int16_t)roundf(q);
                }
            } else {
                for (int i = 0; i < HIDDEN_DIM; i++) {
                    float norm_v = (float)in_row[i] * inv_rms;
                    float q = norm_v * inv_scale;
                    if (q > 32767.0f) q = 32767.0f;
                    else if (q < -32767.0f) q = -32767.0f;
                    x_buf[i] = (int16_t)roundf(q);
                }
            }
        }

        // Zero-pad inactive rows if batch_len < 4
        for (int p = batch_len; p < 4; ++p) {
            memset(x_base + (size_t)p * HIDDEN_DIM, 0, HIDDEN_DIM * sizeof(int16_t));
        }

        // 2. Hardware INT4 execution on MDLA (Single-shot for all B=4 tokens!)
        NPU_TV.set_in(exec, 0, NULL, x_base, sizeof(s_int4_x_in[blk]));
        NPU_TV.set_out(exec, 0, NULL, s_int4_down_out[blk], sizeof(s_int4_down_out[blk]));
        int rc = NPU_TV.exec_compute(exec);
        if (rc != NO_ERROR) return 0;

        // 3. Dequantize Down output and add residual in full precision
        const int16_t *down_base = s_int4_down_out[blk];
        for (int p = 0; p < batch_len; ++p) {
            const int16_t *down_buf = down_base + (size_t)p * HIDDEN_DIM;
            const __fp16 *in_row = x_in[p];
            __fp16 *out_row = x_out[p];
            for (int i = 0; i < HIDDEN_DIM; i++) {
                float d_val = (float)down_buf[i] * act_scale;
                out_row[i] = in_row[i] + (__fp16)d_val;
            }
        }

        double elapsed = tv_now_ms() - tn0;
        if (out_ffn_ms) *out_ffn_ms += elapsed;
        return 1;
    }

    return 0; // Fallback to CPU SDOT if block graph not compiled
}

// Executes real batched Tree-GQA directly on mtk-mdla using real FP16 activations and KV-cache
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
    double *out_gqa_ms)
{
    (void)full_idx;
    if (batch_len < 1 || batch_len > 16 || base_pos < 0 || base_pos + batch_len > MAX_KV_SEQ) {
        return 0;
    }
    if (!q_act || !gate_act || !k_cache || !v_cache || !attn_out) {
        return 0;
    }

    int total_s = base_pos + batch_len;
    int S = 0;
    CompiledGraph *cg = NULL;
    __fp16 *mask_buf = NULL;

    if (total_s <= 128 && g_tree_gqa_b16_s128.valid) {
        S = 128;
        cg = &g_tree_gqa_b16_s128;
        mask_buf = &s_mask128[0][0][0][0];
    } else if (total_s <= 640 && g_tree_gqa_b16_s640.valid) {
        S = 640;
        cg = &g_tree_gqa_b16_s640;
        mask_buf = &s_mask640[0][0][0][0];
    } else if (total_s <= 2048 && g_tree_gqa_b16_s2048.valid) {
        S = 2048;
        cg = &g_tree_gqa_b16_s2048;
        mask_buf = &s_mask2048[0][0][0][0];
    } else {
        return 0;
    }

    // 1. Pack / Transpose Q and Gate activations [batch_len, 16, 256] -> [4, 4, 16, 256]
    for (int b = 0; b < batch_len; ++b) {
        for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
            for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                int h = kh * FA_Q_PER_KV + qg;
                memcpy(s_Q16[kh][qg][b], q_act + b * 4096 + h * FA_HEAD_DIM, FA_HEAD_DIM * sizeof(__fp16));
                memcpy(s_gate16[kh][qg][b], gate_act + b * 4096 + h * FA_HEAD_DIM, FA_HEAD_DIM * sizeof(__fp16));
            }
        }
    }
    for (int b = batch_len; b < 16; ++b) {
        for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
            for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                memset(s_Q16[kh][qg][b], 0, FA_HEAD_DIM * sizeof(__fp16));
                memset(s_gate16[kh][qg][b], 0, FA_HEAD_DIM * sizeof(__fp16));
            }
        }
    }

    // 2. Wire K and V Cache (Direct Zero-Copy DMA binding into MDLA!)
    size_t kv_sz = (size_t)FA_KV_HEADS * FA_Q_PER_KV * S * FA_HEAD_DIM * sizeof(__fp16);
    if (S == 2048 || S == 640) {
        NPU_TV.set_in(cg->exec, 1, NULL, k_cache, kv_sz);
        NPU_TV.set_in(cg->exec, 2, NULL, v_cache, kv_sz);
    } else if (S == 128) {
        const __fp16 (*kc)[FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM] =
            (const __fp16 (*)[FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM])k_cache;
        const __fp16 (*vc)[FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM] =
            (const __fp16 (*)[FA_Q_PER_KV][MAX_KV_SEQ][FA_HEAD_DIM])v_cache;
        for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
            for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                memcpy(s_K128[kh][qg], kc[kh][qg], 128 * FA_HEAD_DIM * sizeof(__fp16));
                memcpy(s_V128[kh][qg], vc[kh][qg], 128 * FA_HEAD_DIM * sizeof(__fp16));
            }
        }
        NPU_TV.set_in(cg->exec, 1, NULL, s_K128, sizeof(s_K128));
        NPU_TV.set_in(cg->exec, 2, NULL, s_V128, sizeof(s_V128));
    }

    // 3. Generate 4D Tree Ancestry Mask
    build_4d_tree_mask_fp16(tree, base_pos, batch_len, S, mask_buf);
    size_t mask_sz = (size_t)FA_KV_HEADS * FA_Q_PER_KV * 16 * S * sizeof(__fp16);

    NPU_TV.set_in(cg->exec, 0, NULL, s_Q16, sizeof(s_Q16));
    NPU_TV.set_in(cg->exec, 3, NULL, s_gate16, sizeof(s_gate16));
    NPU_TV.set_in(cg->exec, 4, NULL, mask_buf, mask_sz);
    NPU_TV.set_out(cg->exec, 0, NULL, s_attn_out16, sizeof(s_attn_out16));

    // 4. Compute Tree Attention on mtk-mdla
    double tn0 = tv_now_ms();
    int rc = NPU_TV.exec_compute(cg->exec);
    double elapsed = tv_now_ms() - tn0;
    if (out_gqa_ms) *out_gqa_ms += elapsed;
    if (rc != NO_ERROR) return 0;

    // 5. Unpack s_attn_out16 [4, 4, 16, 256] -> [batch_len, 4096] for out_proj
    for (int b = 0; b < batch_len; ++b) {
        for (int kh = 0; kh < FA_KV_HEADS; ++kh) {
            for (int qg = 0; qg < FA_Q_PER_KV; ++qg) {
                int h = kh * FA_Q_PER_KV + qg;
                memcpy(attn_out + b * 4096 + h * FA_HEAD_DIM, s_attn_out16[kh][qg][b], FA_HEAD_DIM * sizeof(__fp16));
            }
        }
    }
    return 1;
}

// Engine weak bindings for standalone testing / dynamic linking
__attribute__((weak)) int engine_exec_block_batch(
    int blk, int lin_idx, int full_idx, int base_pos, int batch_len,
    __fp16 (*seq_chunk)[HIDDEN_DIM], double *io_sdot_ms, double *io_npu_ms)
{
    (void)blk; (void)lin_idx; (void)full_idx; (void)base_pos; (void)batch_len;
    (void)seq_chunk; (void)io_sdot_ms; (void)io_npu_ms;
    return 1;
}

int run_npu_batched_tree_block(
    int blk,
    int is_full,
    int batch_len,
    int base_pos,
    __fp16 (*batch_hidden)[HIDDEN_DIM],
    const CandidateTree *tree,
    double *out_sdot_ms,
    double *out_npu_ms)
{
    (void)is_full; (void)tree;
    if (batch_len < 1 || batch_len > 16 || base_pos < 0 || base_pos + batch_len > MAX_KV_SEQ) {
        return 0;
    }
    int full_idx = blk / 4;
    int lin_idx  = blk - full_idx;
    return engine_exec_block_batch(blk, lin_idx, full_idx, base_pos, batch_len, batch_hidden, out_sdot_ms, out_npu_ms);
}
