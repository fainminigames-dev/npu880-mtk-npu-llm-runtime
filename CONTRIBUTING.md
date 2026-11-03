# Contributing

Thanks for interest in NPU 880! This is an independent student research project (WIP).

- PRs with improvements are very welcome (bug fixes, perf, new SoCs, docs).
- Please keep the existing code style (C: gnu11, 4-space indent; Python/shell: simple scripts).
- Add a short note in the PR description about what changed and how you tested it (device + build id).
- Keep the copyright header (`Copyright (c) 2026 Albert — SPDX-License-Identifier: MIT`)
  at the top of source files; new files should carry the same header with your year/name added.
- Do NOT add: model files (*.gguf), vendor binaries/libraries, firmware, probe dumps,
  logs with private paths, or any MediaTek proprietary material (see DISCLAIMER.md).

## Reporting issues
Open a GitHub issue with: device (SoC), Android/HyperOS version, build id from the log
(`ENGINE_BUILD_ID`), prompt, and the relevant log lines (`SESSION_BENCHMARK`, `NPU_ERR`, ...).
