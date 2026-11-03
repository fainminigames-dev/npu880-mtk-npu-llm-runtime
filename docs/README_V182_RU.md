# V182-R2: Отчёт об аудите сбоя V182 и исправленный NPU Runtime (`build_id = v182-r2-20261027-audit`)

## 1. Точная причина сбоя V182 (почему после `Hello` возвращались `answer=""` и `timings: 0.00`)

1. **Переполнение памяти ION/RAM (`rc = 4` `ANEURALNETWORKS_OUT_OF_MEMORY`) на первом же шаге `B = 1` decode (`g = 1`)**:
   - В `v182` макрос `#define PREFILL_BATCH` остался равным `28` (строка 100), при этом одновременно компилировались все графы `B = 28` и все 4 новых графа `B = 1` Full-Attention (`fa_frontend_all`, `fullattn_qknorm_rope`, `fa_gqa_only`, `out_proj_4096_2560_res`), а функция `init_pinned_lm_and_block_caches()` принудительно выделяла до **`2.08 GB` анонимной RAM** (`363.8 MB` под 3 чанка `LM Head` + до `8 × 215.3 MB` под блоки).
   - На первом запросе `"hello!"` префилл (`run_prompt_prefill_block_major_pipelined`) и первый вызов `LM Head` (`token_index: 0` -> `"Hello"`) успевали пройти.
   - Но на шаге `g = 1` (`run_one_token_32blocks_pipelined`) драйвер `libneuronusdk_adapter.mtk.so` при первом вызове `NeuronExecution_compute()` на графах `B = 1` пытался выделить рабочие ION/DMA-буферы в ядре и получал отказ по памяти (`rc = 4`).
2. **Маскировка ошибки в `run_chatml_prompt_session_ex()` и `http_server_thread_fn()`**:
   - В начале `run_chatml_prompt_session_ex()` структура `g_last_session` обнулялась (`memset(&g_last_session, 0, ...)`), а заполнялась только в самом конце функции после цикла `for (; g < max_new_tokens; ++g)`.
   - Когда `run_one_token_32blocks_pipelined()` возвращал `0` на `g = 1`, `run_chatml_prompt_session_ex()` немедленно делал `return 0`, оставляя `g_last_session` полностью нулевой (`answer=""`, `ttft_ms=0.00`, `prefill_ms=0.00`, `decode_ms=0.00`) и **не вызывая `wait_async_prefetch(NULL, NULL)`**.
   - А `http_server_thread_fn()` игнорировал возвращаемое значение `0` и отправлял клиенту фиктивный финальный чанк с `finish_reason="stop"`, пустым `answer=""`, нулевыми таймингами и `data: [DONE]`.
3. **Почему падали все следующие запросы (`"hy!"`, `"whats youre name?"`, повторный `"hello!"`)**:
   - В `run_graph()` при ошибке `NPU.execCompute(ex) != NO_ERROR` сломанный дескриптор `cg->persistent_exec` не освобождался (`NPU.execFree`) и не сбрасывался в `NULL`.
   - Кроме того, при раннем `return 0` фоновая задача `Prefetcher` оставалась незавершённой, а на блоке 31 при нечётном `first_stream_blk` (`g_num_pinned_blocks % 2 == 1`) префетчер начинал перезаписывать `g_pipe_slot[1]` в тот момент, когда `Block 31` (`31 % 2 == 1`) ещё читал веса из `g_pipe_slot[1]`.

## 2. Что исправлено в `v182-r2` (`build_id = v182-r2-20261027-audit`)

1. **Идентичность бинарника, SHA-256, `/health` и безопасная работа с PID**:
   - Введён `ENGINE_BUILD_ID = "v182-r2-20261027-audit"`.
   - Добавлен эндпоинт `GET /health` (и `GET /v1/models`), возвращающий `build_id`, `version`, `pid`, `prefill_batch` (`16`), `max_kv_seq` (`128`), объём свободной RAM при старте и сейчас, точные размеры выделенных буферов в байтах, статус верификации каждого графа и последнюю ошибку NPU.
   - Скрипт `run_v182_turbo_pinned_ram_pure_npu_engine.ps1` вычисляет локальный `SHA-256` скомпилированного бинарника, сверяет его с `sha256sum` на устройстве и проверяет `GET /health` (`status == "ok"` и совпадение `build_id`) перед отправкой запроса на тёплый сервер. Широкие `pkill -9 -f qwen35` убраны — останавливается только проверенный PID `pidof v182_turbo_pinned_ram_pure_npu_engine`.
2. **Честная обработка ошибок NPU, HTTP и SSE**:
   - Добавлены категории ошибок (`GRAPH_COMPILE_ERROR`, `INPUT_BIND_ERROR`, `OUTPUT_BIND_ERROR`, `EXECUTION_COMPUTE_ERROR`, `RUNTIME_LOGIC_ERROR`, `KV_CACHE_OVERFLOW_ERROR`) с записью этапа (`PREFILL`, `DECODE_BLOCK`, `LM_HEAD`), имени графа, `npu_rc`, номера блока, позиции токена и шага генерации.
   - При ошибке `NeuronExecution_compute()` повреждённый `persistent_exec` немедленно освобождается, `wait_async_prefetch` корректно дожидается фонового потока, а слоты `g_pipe_slot[0..1].blk_idx` сбрасываются.
   - При ошибке сервер возвращает HTTP `500` (в JSON-режиме) или событие `chat.completion.error` с `finish_reason="error"` (в SSE-режиме) с фактическими частичными таймингами и **не отправляет** фиктивный `finish_reason="stop"` или `data: [DONE]`.
3. **Аппаратный `NeuronExecution_compute` Probe при старте и атомарное обновление состояния (`B = 1` vs `Fallback`)**:
   - При старте функция `probe_all_compiled_graphs_on_npu()` реально вызывает `NeuronExecution_compute()` на `B = 1` Linear Attention, `B = 1` Full-Attention, `B = 16` Batched и `LM Head`. Если любой опциональный `B = 1` граф возвращает `rc = 4`, его память ION немедленно освобождается при старте.
   - Путь выполнения (`B1_NATIVE` или `B16_FALLBACK`) выбирается **до** запуска блока; повторного запуска упавшего блока поверх изменённого состояния больше нет.
   - В `exec_single_block_one_token_npu()` запись в `state->conv_hist` и `state->gdr_state` перенесена в самый конец функции (после успешного завершения `ffn_sublayer_all`), а при ошибке в Full-Attention слот `k_cache`/`v_cache` для текущей позиции очищается.
4. **Устранены все несоответствия исходника и README**:
   - `#define PREFILL_BATCH 16` и `#define MAX_KV_SEQ 128` установлены в исходнике.
   - По умолчанию `pin_lm = 0` и `pin_blocks = 0` (консервативный бюджет RAM без риска `rc = 4`; включается явно флагами `-pinlm` / `-pinblocks` для контролируемых замеров).
   - Проверен токен `25677` по словарю GGUF (`VOCAB_TOKEN_AUDIT` при старте): `25677` (`' ðŁĺ'`) + `232` (`'Ĭ'`) — это первые 3 байта и 4-й байт UTF-8 эмодзи `😊` (`0xF0 0x9F 0x98 0x8A`) в таблице GPT-2 Byte-Level BPE, а не EOS. Реализован полный декодер GPT-2 Byte-Level BPE -> UTF-8 и сохранение корректного UTF-8 в `json_escape_str` и SSE.
