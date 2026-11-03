# Контрольная точка проекта MTK NPU / Qwen3.5-4B

**Дата:** 12 октября 2026 г.  
**Язык проекта:** русский  
**Цель:** изучить возможность выполнять вычисления линейного внимания Qwen3.5-4B на MediaTek MDLA через NeuroPilot SDK без root-доступа, начиная с небольших воспроизводимых C-тестов и постепенно подключая реальные веса GGUF.

> Это документ передачи контекста для нового чата. Он фиксирует подтверждённые результаты и явно отделяет их от того, что ещё не сделано. Не считать предыдущие синтетические тесты полноценным запуском Qwen.

---

## 1. Среда

### Телефон и NPU
- Телефон: **POCO X7 Pro**.
- ОС: Android 16.
- ABI: `arm64-v8a`.
- SoC/платформа: MediaTek MT6899.
- NeuroPilot adapter: `libneuronusdk_adapter.mtk.so`.
- `Neuron_getVersion`: **8.2.26**.
- `Neuron_getDeviceCount`: 3 устройства: `mtk-gpu`, `mtk-dsp`, `mtk-mdla`.
- Root не используется.
- ADB ранее подключался с device ID `PVGAY9RSZTU86PZD`; после отключения телефона соединение было восстановлено.

### Ноутбук и инструменты
- Windows PowerShell.
- Android NDK: `C:\Android\android-ndk-r30`.
- Clang: `C:\Android\android-ndk-r30\toolchains\llvm\prebuilt\windows-x86_64\bin\clang.exe`.
- Типичная команда сборки:
  ```powershell
  & $clang --target=aarch64-linux-android24 -std=gnu11 -O2 .\source.c -o .\program -ldl -lm
  ```
- ADB используется для загрузки C-бинарников и небольших тензоров на телефон.

### Файл модели на телефоне
`/storage/emulated/0/models/Qwen3.5-4B-Q5_K_M.gguf`

Подтверждённый размер: **3,143,656,608 байт = 2.928 GiB**. Полный GGUF не копировался на ноутбук.

---

## 2. Результат GGUF-аудита v10.1

Из файла на телефоне был считан небольшой префикс GGUF, достаточный для разбора заголовка и каталога тензоров.

- Формат: GGUF v3.
- Architecture: `qwen35`.
- Model name: `Qwen3.5-4B`.
- Metadata: 46 записей.
- Tensor count: 426.
- Blocks/layers: 32.
- Hidden size: 2560.
- FFN size: 9216.
- Full-attention interval: 4.
- По этой metadata-интерпретации: 24 блока linear attention и 8 блоков full attention.
- Типы тензоров по каталогу: F32 177, Q5_K 131, Q6_K 70, Q8_0 48.

Репрезентативные реальные тензоры block 0 из каталога:
- `blk.0.attn_qkv.weight`: GGUF dims `2560x8192`, Q6_K.
- `blk.0.attn_norm.weight`: 2560, F32.
- `blk.0.ssm_conv1d.weight`: `4x8192`, F32.
- `blk.0.ssm_alpha.weight`: `2560x32`, Q8_0.
- `blk.0.ssm_beta.weight`: `2560x32`, Q8_0.
- `blk.0.attn_gate.weight`: `2560x4096`, Q5_K.
- `blk.0.ssm_a`: 32, F32.
- `blk.0.ssm_dt.bias`: 32, F32.
- `blk.0.ssm_norm.weight`: 128, F32.
- `blk.0.ssm_out.weight`: `4096x2560`, Q6_K.
- `blk.0.ffn_gate.weight`: `2560x9216`, Q5_K.
- `blk.0.ffn_up.weight`: `2560x9216`, Q5_K.
- `blk.0.ffn_down.weight`: `9216x2560`, Q6_K.

У full-attention блока, например block 3, каталог содержит `attn_q`, `attn_k`, `attn_v`, `attn_output`, q/k norms и соответствующие FFN-тензоры.

Полезный локальный артефакт исходного аудита был создан в папке:
`mtk_qwen35_gguf_audit_v10_1_adb_prefix`.

---

## 3. Реальные веса: v11.1 — успешный пробник QKV

Успешно извлечён **один настоящий тензор** `blk.0.attn_qkv.weight` с телефона, не перенося весь 2.9-GiB GGUF.

- Тензор: Q6_K.
- GGUF dimensions: `2560x8192`.
- Абсолютный offset данных: `539,669,664`.
- Packed tensor payload: `17,203,200` байт.
- Выравненный диапазон извлечения: `17,207,296` байт.
- Число элементов: `20,971,520`.
- После деквантования: FP16, `41,943,040` байт.
- SHA-256 raw tensor: `a17214adbc34cc9dec62df6e3890ee0863ce57197aa9155c477a7a25373149f3`.
- SHA-256 FP16 tensor: `915887008071e146a4bda116f0707396beecc7ee5caab67bd4847de98f29f4fb`.
- RMS весов после преобразования: `0.013224007`.
- Максимальный |weight|: `0.39038086`.
- Ошибка FP16 cast относительно FP32 dequant: max `0.00012207031`, mean `1.6502337e-06`.

На телефоне MDLA успешно скомпилировал и выполнил FC с этими реальными QKV-весами:
- Activation и bias в этом конкретном v11.1 тесте синтетические (bias нулевой).
- Graph: `[1,2560] × real weight [8192,2560]`.
- `compile: OK`.
- Max absolute error MDLA vs CPU при одних и тех же FP16 input/weight buffers: `0.00012207` (output index 8076).
- Mean absolute error: `0.00000005`.
- Среднее время графа: `1.3699 ms` (warmup 2, измерений 10; только этот FC).

Это доказало, что извлечение одного Q6_K-тензора, его деквантование в FP16 и использование его как константы FC на MDLA работают. Это не было inference.

Исходный комплект назывался `mtk_qwen35_real_tensor_probe_v11_1_fix`.

---

## 4. Реальная активация + реальная RMSNorm + реальные QKV-веса: v12.1

Первый вариант v12 остановился на скриптовой ошибке: аудит не содержал `tokenizer.ggml.bos_token_id`. В v12.1 скрипт принял явно выбранный токен **`<|im_start|>`**, ID `248045`, как первый токен chat-template, поскольку configured BOS token отсутствовал.

В v12.1 были извлечены:

### Token embedding row
- Tensor: `token_embd.weight`.
- Тип: Q6_K.
- GGUF dims: `2560x248320`.
- Row ID: `248045` (`<|im_start|>`).
- Absolute offset row: `531,872,964`.
- Row raw payload: `2,100` байт (extract был выровнен до 8,192 байт).
- После Q6_K -> FP16: 2,560 элементов, 5,120 байт.
- SHA-256 FP16 row: `d36d8e5a27ae5487a17ddccd056fc3727a79ef0c2a2085a7cc92e395027afbff`.
- Max abs embedding value: `0.045410156`; RMS `0.010639415`.

### Block-0 RMSNorm gamma
- Tensor: `blk.0.attn_norm.weight`.
- Тип источника: F32.
- Offset: `539,659,424`.
- Source bytes: `10,240`; converted FP16 bytes `5,120`.
- SHA-256 FP16: `2528e22cd200bd3a5f1ab69aab66f71f23b7893ce7ad8998185d040a5ef8da24`.

### MDLA graph v12.1
- Real token embedding `[1,2560]` + real gamma + real QKV weight.
- Graph: RMSNorm `[1,2560]` -> FC `[2560->8192]`, all on `mtk-mdla`.
- RMSNorm epsilon `1e-6`.
- `compile: OK`.
- RMSNorm max/mean error vs CPU: `0 / 0`.
- QKV output max absolute error vs CPU with identical FP16 buffers: `0.00781250` at index `5862`; mean absolute error `0.00000137`.
- First 8 CPU and MDLA QKV outputs printed identical to shown precision: `-0.1556396 0.0750122 0.2670898 2.0058594 -1.0615234 1.7177734 -0.7851562 -0.2176514`.
- Average graph time: `1.5624 ms` (warmup 2, 10 runs; only real embedding + RMSNorm + QKV FC graph).

**Latest confirmed state:** one real token embedding, real block-0 `attn_norm` gamma, and real `attn_qkv.weight` have been used together in one MDLA graph. Still not full model inference. Causal convolution and the `ssm_alpha`/`ssm_beta`/`attn_gate`/`ssm_a`/`ssm_dt.bias`/`ssm_norm`/`ssm_out` are not yet wired in with real weights.

Folder on Windows:
`C:\Users\alikK\mtk-npu-project\mtk_qwen35_real_first_token_v12_1_chat_start`

---

## 5. Синтетический путь v1–v9: что реально подтверждено

Все тесты ниже использовали **синтетические веса**, если в описании не сказано иное.

### Доступ к NeuroPilot / MDLA
- `dlopen("libneuronusdk_adapter.mtk.so")` succeeds inside Android program.
- NeuroPilot version `8.2.26`.
- Device enumeration returned `mtk-gpu`, `mtk-dsp`, `mtk-mdla`.
- FP16/INT8 FC test graphs compile and run on MDLA.
- Некоторые конкретные тестовые графы на GPU и DSP возвращали compile rc=6; это не доказывает общую несовместимость GPU/DSP.

### QKV + Conv1D path
Подтверждена работа на MDLA:
- RMSNorm over hidden width 2560.
- FC projection `2560 -> 8192`.
- reshape to NHWC `[1,1,T,8192]`.
- causal depthwise Conv1D kernel 4 with left padding 3, right padding 0.
- SiLU.
- `STRIDED_SLICE` splitting Q/K/V shapes corresponding to `[1,1,T,2048]`, `[1,1,T,2048]`, `[1,1,T,4096]`.
- Additional synthetic branches `2560->32` for `a/b` and `2560->4096` for `z`.

For a synthetic 2-token graph, an example max absolute CPU/NPU error for final SiLU was `0.00001144`; timing is graph/shape-specific and should not be represented as model token speed.

### Q/K normalization + Gated Delta Rule
- Q/K normalized over 16 key heads × 128 dimension; Q additionally scaled by `1/sqrt(128)`.
- Linear-attention recurrence in the probe: 16 key heads -> 32 value/state heads, group=2, K=V=128.
- `beta = sigmoid(beta_logits)`.
- `a + dt_bias` followed by approximate softplus, then decay `exp(-exp(A_log)*softplus(...))` in the test implementation.
- Synthetic parameters include `A_log`, `dt_bias`, initial state, and weights unless otherwise noted.
- Two-token synthetic recurrent test compiled and ran on MDLA. Reported max error across outputs/final states was `0.00009155`; a later beta/a projected test reported `0.00006104`.
- Approximate softplus max errors seen were around `0.0007–0.0015`; decay max error around `0.00048828`. These are FP16 test results, not an exact reproduction claim.

### Gated RMSNorm / output / residual
- Per-head RMSNorm over 128 values.
- Multiply by gamma and `SiLU(z)`.
- Reshape `[T,32,128] -> [T,4096]`.
- FC `4096 -> 2560`.
- Residual ADD with original hidden vector.
- All tested successfully as an MDLA graph with synthetic weights.
- One v4 residual probe had max sum error `0.00001526`, mean `0.00000009`; in later per-token runs small errors varied with inputs.

### Stateful continuation v5.1
- Recurrent state size: `524,288` FP16 elements.
- State save/load round-trip: bit exact, max diff `0`.
- With fresh `NeuronExecution` and identical restored starting state for CPU and MDLA, continuation max error returned to `0.00009155`, mean `0.00000147`.
- Earlier v5 diagnostic had a false large error `0.01427460` due to a flawed continuation test; v5.1 fixed comparison conditions/execution handling.

### Causal Conv1D cache v6 / v6.1
- History: last 3 pre-convolution vectors, shape `[1,1,3,8192]`, 24,576 FP16 values.
- One-token graph concatenates history `[1,1,3,8192]` with current projection `[1,1,1,8192]`, producing `[1,1,4,8192]`, then valid depthwise Conv1D + SiLU.
- MDLA CONCATENATION compile/run succeeded.
- For each of 3 tokens, streamed output vs full MDLA sequence graph max error `0.00000000`; vs CPU max error `0.00001144`.
- Cache save/load after every token: bit-exact.

### Integrated streaming v7 / v8 / v9
- v7: Conv1D cached streaming output feeds per-token Q/K/V to Gated Delta Rule; GDR state persisted per token. GDR max error outputs/state typically `0.00006104`.
- v8: added per-token gated RMSNorm, output projection, and residual add.
- v9: additionally computes RMSNorm and QKV / a / b / z projections separately for each single-token input; 3 one-token projection invocations matched batch reference exactly in logged FP16 output: max differences `0` for QKV, beta, a, and z.
- v9 still used pre-generated synthetic hidden vectors and synthetic weights. The pipeline was a streaming computation test, not generated autoregressive text.

---

## 6. Important caveats / interpretation

1. **No full Qwen3.5-4B inference has been demonstrated.** No real text logits, next-token sampling, prompt processing across all 32 blocks, or output text generation have been produced by this prototype.
2. v1–v9 tested architecture-like graphs using synthetic weights and synthetic token hidden states.
3. v11.1 used one real QKV matrix with synthetic activation and zero bias.
4. v12.1 used one real selected token embedding, real block-0 RMSNorm gamma, and real QKV matrix in the same graph. Later block-0 operations still remain to be replaced with actual weights.
5. All fully integrated stream timings are per-test-graph, not overall tokens/sec. Runtime also includes host orchestration / handoff if reported outside the MDLA-only benchmark. Don’t sum benchmark snippets and call it model throughput without a controlled end-to-end run.
6. Error is generally max absolute and mean absolute error vs CPU using the same FP16 buffers; FP16 accumulation/rounding may differ, so max and mean may tell different stories. The v12.1 FC max error `0.0078125` is higher than the tiny means and must be investigated across more activations before claiming exact equivalence.
7. Qwen3.5 hybrid architecture requires both linear-attention and full-attention blocks. The current streaming chain covers only an architecture-like linear-attention path, not all required parts of the model.
8. Keep model files on the phone; only extract selected tensor byte ranges. Don’t transfer the entire multi-GB GGUF unless it becomes necessary.

---

## 7. Current folders and useful artifacts

On Windows, project root:
`C:\Users\alikK\mtk-npu-project`

Latest completed project folders:
- `mtk_npu_conv_gdr_stream_v9_per_token_proj` — all synthetic per-token projections, Conv cache, recurrent state, per-token gated output/residual.
- `mtk_qwen35_gguf_audit_v10_1_adb_prefix` — partial GGUF inventory audit and JSON.
- `mtk_qwen35_real_tensor_probe_v11_1_fix` — raw Q6_K block-0 QKV tensor extraction/dequantization and MDLA FC probe.
- `mtk_qwen35_real_first_token_v12_1_chat_start` — actual `<|im_start|>` embedding + actual attention RMSNorm gamma + actual QKV projection on MDLA.

Phone model path:
`/storage/emulated/0/models/Qwen3.5-4B-Q5_K_M.gguf`

Common paths written by earlier test binaries in `/data/local/tmp` are temporary handoff/cache/state files and can be cleared by later scripts:
- `/data/local/tmp/mtk-npu-qkv-stage.bin`
- `/data/local/tmp/mtk-npu-gdr-state.bin`
- `/data/local/tmp/mtk-npu-gdr-output.bin`
- `/data/local/tmp/mtk-npu-gdr-output-token-*.bin`
- `/data/local/tmp/mtk-npu-hidden-token-*.bin`

Only clear paths belonging to a specific test when that test script requests it; don’t casually erase the model under `/storage/emulated/0/models/`.

---

## 8. Recommended next steps (real weights, incremental)

Priority is now to extend the **actual block-0 path**, not to make another synthetic demo:

1. **Audit shapes / GGUF layout carefully.** Keep using partial reads of the GGUF tensor directory to calculate raw data offsets. Validate each extraction by exact expected byte counts and checksums.
2. **Replace the synthetic Conv1D kernel:** extract real `blk.0.ssm_conv1d.weight` (F32, dims `4x8192`) and add it to a graph that consumes real QKV projection output. Confirm channel and sequence layout and compare MDLA/CPU outputs.
3. **Extract real alpha/beta projection tensors:** `blk.0.ssm_alpha.weight` and `blk.0.ssm_beta.weight` are Q8_0 and dims `2560x32`. Implement and unit-test Q8_0 dequantization, and probe these actual matrices with the real normalized hidden activation.
4. **Extract actual `blk.0.attn_gate.weight`** (Q5_K, 2560x4096), dequantize and test `z` projection.
5. Extract actual parameters `blk.0.ssm_a`, `blk.0.ssm_dt.bias`, `blk.0.ssm_norm.weight`, and `blk.0.ssm_out.weight`; test each independently before connecting the actual one-token block.
6. Maintain a CPU reference using exactly the same dequantized FP16 weight and activation buffers passed into MDLA. Log max error, mean error, RMS error, and worst index for each operation.
7. Only after block-0 linear attention matches with actual weights should the code start connecting the actual residual/FFN/post-attention norm path.
8. Full model is more involved because 8 blocks require full attention and all layers require actual FFN weights, embedding/head and tokenizer/template handling. Don’t call block-0 success a complete model.

The next best isolated test is **real block-0 Conv1D weights + real block-0 QKV output for the selected token**, before attempting the entire recurrent block with all authentic parameters.

---

## 9. Handover message for a new chat

Paste this short message along with this checkpoint if the current chat stops accepting turns:

> Continue my MTK NPU project from the attached checkpoint. I use a POCO X7 Pro (Android 16, MediaTek MT6899, NeuroPilot 8.2.26, no root) and Windows PowerShell/Android NDK r30. I have already made synthetic MDLA tests for RMSNorm, QKV FC, Conv1D + SiLU with persisted 3-frame cache, Q/K normalization, two-token / one-token Gated Delta Rule with recurrent state persistence, gated RMSNorm, out_proj and residual. v9 streaming version works with synthetic inputs/weights. I then audited `/storage/emulated/0/models/Qwen3.5-4B-Q5_K_M.gguf` (GGUF v3, 32 layers, hidden 2560, FFN 9216, 426 tensors). v11.1 extracted real `blk.0.attn_qkv.weight` Q6_K, dequantized it to FP16 and ran the MDLA FC successfully. v12.1 ran the actual `<|im_start|>` token embedding (ID 248045), real `blk.0.attn_norm.weight` gamma and real QKV matrix together in an MDLA graph: RMSNorm max/mean error 0, QKV vs CPU max abs error 0.0078125 (index 5862), mean 0.00000137, average FC graph time 1.5624 ms. This is not full inference. Next task: extract the real `blk.0.ssm_conv1d.weight` F32 tensor and test it with the real token's QKV outputs; then gradually replace actual alpha/beta/z/A/dt/norm/out weights. Preserve caveats and don’t claim complete Qwen inference until logits/text are actually generated.
