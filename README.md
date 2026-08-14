# Running Cogni-Brain on DGX Spark · Nemotron-3.5-Lightning-30B-A3B-NVFP4 + DSpark

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Base Model](https://img.shields.io/badge/base%20model-Nemotron--3.5--Lightning--30B--A3B--NVFP4-cyan)
![Speculative](https://img.shields.io/badge/speculative-DSpark--NVFP4%20(3%20tokens)-purple)
![Runtime](https://img.shields.io/badge/runtime-vLLM%20%2F%20vllm--openai-orange)
![Hardware](https://img.shields.io/badge/hardware-NVIDIA%20DGX%20Spark-brightgreen?logo=nvidia&logoColor=white)
![Context](https://img.shields.io/badge/context-1M-blue)
![Tool Calling](https://img.shields.io/badge/tool--calling-qwen3__coder-green)
![Reasoning](https://img.shields.io/badge/reasoning-nemotron__v3-black)
![Quantization](https://img.shields.io/badge/quantization-NVFP4-purple)

This repo documents my inference tuning experiments of [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) with DSpark speculative decoding on a single DGX Spark.

> ⚠️ **Personal workstation setup. Not for enterprise use. Use at your own risk.**

---

## What is Nemotron 3.5 Lightning?

[NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) is NVIDIA's 30B parameter dense model, served here as **Cogni-Brain** — a model alias that stays consistent across agent frameworks (Claude Code, Continue, Open WebUI, etc.) regardless of which model is running underneath.

Key characteristics:
- **NVFP4** — Leverages native FP4 Tensor Cores on the DGX Spark
- **DSpark Speculative Decoding** — `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` generates **3 draft tokens per step** (num_speculative_tokens: 3) for significant TPS gains
- **nemotron_v3** reasoning parser and **qwen3_coder** tool-call parser
- **1M token context window** (`--max-model-len 1048576`)
- Native `--enable-auto-tool-choice` support

---

## Quick Start

> ⚠️ **Warning:** `setup/install.sh` disables system swap permanently to prevent unified-memory thrashing. This is a system-level change that survives reboots.

```bash
# 1. Set your Hugging Face token (model may be gated)
export HF_TOKEN="your_hf_token_here"

# 2. System prerequisites (swap disable, uv install, docker check)
bash setup/install.sh

# 3. Download model weights (one-time)
bash setup/download_model.sh

# 4. Preflight check — validates GPU, swap, memory, Docker image, flag compat
bash setup/preflight.sh

# 5. Start vLLM container with NVFP4 + DSpark
bash docker/start.sh

# 6. Check container status & logs
bash docker/status.sh
```

---

### 5. Run Performance Benchmarks

Run the benchmark scripts to test baseline latency, tool-calling smarts, and spark-arena-style llama-benchy sweeps:

```bash
# 1. Custom Speed Benchmark (TPS, TTFT, concurrency, 1M context limits)
uv run benchmark/benchmark_speed.py

# 2. Smarts Benchmark (tool-eval-bench)
uv run benchmark/benchmark_smarts.py

# 3. Full spark-arena-style overnight sweep (llama-benchy)
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
```

> `benchmark_speed_arena.py` and `benchmark_smarts.py` fetch `llama-benchy` and `tool-eval-bench` through `uv` on demand — no separate global install step required.

---

## Benchmarks

### Speed Benchmark (custom script)

**August 2026 Results (vLLM v0.27.1 + CUDA Graphs):**
- **Baseline TPS:** ~44.2 tokens/sec (Peak: 45.7 tok/s)
- **Time to First Token (TTFT):** ~69 ms (steady state)
- **Max Throughput (4 sessions):** 98.8 tokens/sec total (24.7 tok/s per session)
- **Deep Context Scaling Throughput:** 142.0 tok/s at ~523K context
- **Max Working Context:** ~522,986 tokens

![Speed Benchmark Results](assets/benchmark_speed_august2026.png)

```bash
# Full run (TPS, TTFT, concurrent sessions, context window)
uv run benchmark/benchmark_speed.py

# Skip slower subtests for a quicker check
uv run benchmark/benchmark_speed.py --skip-context
uv run benchmark/benchmark_speed.py --skip-context --skip-concurrent

# Custom endpoint or model alias
uv run benchmark/benchmark_speed.py --host localhost --port 8000 --model Cogni-Brain
```

> Tests: baseline TPS, TPS vs output length, concurrent sessions (1–4), context window (up to 1M, the server's `--max-model-len`), health & KV stats.

### Smarts Benchmark (tool-eval-bench)

**August 2026 Results:**
- **Final Score:** 80 / 100 (**** Good)
- **Pass Rate:** 11 Passed, 2 Partial, 2 Failed
- **Responsiveness:** Median turn 3.4s
- **Efficiency:** 0.5 pts/1K tokens

![Smarts Benchmark Results 1](assets/benchmark_smarts_august2026_1.png)
![Smarts Benchmark Results 2](assets/benchmark_smarts_august2026_2.png)

```bash
# Quick smoke test (recommended first run)
uv run benchmark/benchmark_smarts.py

# Throughput sweep
uv run benchmark/benchmark_smarts.py --mode perf

# Deterministic multi-trial evaluation
uv run benchmark/benchmark_smarts.py --mode trials --seed 42 --trials 3
```

> Evaluates: tool selection, parameter precision, multi-step chains, refusal behaviour, error recovery.

### spark-arena Benchmark (overnight, llama-benchy)

```bash
# Standard spark-arena sweep (tests context depths up to 1M)
uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv

# Custom endpoint
uv run benchmark/benchmark_speed_arena.py \
  --base-url http://localhost:8000/v1 \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 \
  --served-model-name Cogni-Brain \
  --save-result benchmark/results_full.csv
```

> This sweep tests concurrency 1, 2, 5, 10 at depth points up to 1M tokens: `0`, `4096`, `8192`, `16384`, `32768`, `65535`, `131072`, `262144`, `524288`, `1048576`.
> Plan for several hours; run overnight.

**August 2026 Results (vLLM v0.27.1 + CUDA Graphs Sweep):**

> **Note:** `ctx_*` tests load the context as a long system prompt before running the benchmark. `pp2048` tests inject all tokens as a prompt prefix. Results reflect vLLM v0.27.1 with CUDA Graphs enabled across context depths up to 512K (`d524288`).

| test | t/s (total) | t/s (req) | peak t/s | peak t/s (req) | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| pp2048 (c1) | 5756.67 ± 299.73 | 5756.67 ± 299.73 | | | 358.91 ± 19.19 | 356.70 ± 19.19 | 358.91 ± 19.19 |
| tg128 (c1) | 41.76 ± 0.72 | 41.76 ± 0.72 | 44.00 ± 0.82 | 44.00 ± 0.82 | | | |
| pp2048 (c2) | 6039.83 ± 11.81 | 4218.51 ± 1188.31 | | | 529.53 ± 148.52 | 527.32 ± 148.52 | 529.53 ± 148.52 |
| tg128 (c2) | 59.52 ± 0.57 | 30.97 ± 1.10 | 67.33 ± 0.47 | 34.00 ± 0.58 | | | |
| pp2048 (c5) | 6313.96 ± 26.03 | 2307.50 ± 1555.13 | | | 1188.40 ± 466.79 | 1186.19 ± 466.79 | 1188.40 ± 466.79 |
| tg128 (c5) | 86.56 ± 0.52 | 19.78 ± 1.38 | 114.67 ± 1.70 | 23.13 ± 0.62 | | | |
| pp2048 (c10) | 6554.46 ± 3.13 | 1512.12 ± 1336.54 | | | 1987.49 ± 866.98 | 1985.28 ± 866.98 | 1987.49 ± 866.98 |
| tg128 (c10) | 113.69 ± 3.03 | 13.88 ± 1.39 | 177.67 ± 2.87 | 18.10 ± 0.60 | | | |
| ctx_pp @ d4096 (c1) | 5917.47 ± 40.66 | 5917.47 ± 40.66 | | | 694.60 ± 4.75 | 692.39 ± 4.75 | 694.60 ± 4.75 |
| ctx_tg @ d4096 (c1) | 41.83 ± 0.68 | 41.83 ± 0.68 | 43.67 ± 0.47 | 43.67 ± 0.47 | | | |
| pp2048 @ d4096 (c1) | 1976.43 ± 7.31 | 1976.43 ± 7.31 | | | 1038.44 ± 3.84 | 1036.23 ± 3.84 | 1038.44 ± 3.84 |
| tg128 @ d4096 (c1) | 41.42 ± 0.64 | 41.42 ± 0.64 | 44.00 ± 0.00 | 44.00 ± 0.00 | | | |
| ctx_pp @ d4096 (c2) | 6237.80 ± 93.43 | 3281.18 ± 166.95 | | | 1254.08 ± 63.53 | 1251.87 ± 63.53 | 1254.08 ± 63.53 |
| ctx_tg @ d4096 (c2) | 56.22 ± 7.33 | 32.15 ± 3.49 | 64.33 ± 3.77 | 34.74 ± 4.84 | | | |
| pp2048 @ d4096 (c2) | 2098.47 ± 16.11 | 1143.49 ± 93.85 | | | 1805.34 ± 147.75 | 1803.13 ± 147.75 | 1805.34 ± 147.75 |
| tg128 @ d4096 (c2) | 58.11 ± 1.30 | 30.24 ± 1.02 | 66.00 ± 0.82 | 33.67 ± 1.11 | | | |
| ctx_pp @ d4096 (c5) | 6442.31 ± 35.31 | 2058.03 ± 791.63 | | | 2263.14 ± 729.68 | 2260.93 ± 729.68 | 2263.14 ± 729.68 |
| ctx_tg @ d4096 (c5) | 75.05 ± 2.33 | 18.86 ± 2.78 | 110.67 ± 1.70 | 23.93 ± 1.48 | | | |
| pp2048 @ d4096 (c5) | 2137.26 ± 4.38 | 704.26 ± 281.19 | | | 3367.31 ± 1179.99 | 3365.10 ± 1179.99 | 3367.31 ± 1179.99 |
| tg128 @ d4096 (c5) | 68.41 ± 0.41 | 17.38 ± 2.73 | 112.33 ± 2.05 | 23.07 ± 0.77 | | | |
| ctx_pp @ d4096 (c10) | 6418.42 ± 0.83 | 1421.23 ± 838.63 | | | 3751.53 ± 1645.41 | 3749.32 ± 1645.41 | 3751.53 ± 1645.41 |
| ctx_tg @ d4096 (c10) | 89.42 ± 4.71 | 12.06 ± 2.41 | 167.00 ± 2.83 | 17.53 ± 1.20 | | | |
| pp2048 @ d4096 (c10) | 2122.58 ± 4.81 | 476.15 ± 300.99 | | | 5808.10 ± 2649.36 | 5805.89 ± 2649.36 | 5808.10 ± 2649.36 |
| tg128 @ d4096 (c10) | 77.30 ± 0.20 | 11.24 ± 2.56 | 167.00 ± 1.63 | 17.80 ± 1.11 | | | |
| ctx_pp @ d8192 (c1) | 5853.64 ± 11.84 | 5853.64 ± 11.84 | | | 1401.86 ± 2.83 | 1399.65 ± 2.83 | 1401.86 ± 2.83 |
| ctx_tg @ d8192 (c1) | 42.36 ± 0.46 | 42.36 ± 0.46 | 44.00 ± 0.82 | 44.00 ± 0.82 | | | |
| pp2048 @ d8192 (c1) | 1173.48 ± 1.66 | 1173.48 ± 1.66 | | | 1747.44 ± 2.47 | 1745.23 ± 2.47 | 1747.44 ± 2.47 |
| tg128 @ d8192 (c1) | 41.27 ± 1.52 | 41.27 ± 1.52 | 43.00 ± 0.82 | 43.00 ± 0.82 | | | |
| ctx_pp @ d8192 (c2) | 6149.91 ± 15.37 | 3579.21 ± 501.71 | | | 2337.14 ± 327.30 | 2334.93 ± 327.30 | 2337.14 ± 327.30 |
| ctx_tg @ d8192 (c2) | 50.22 ± 7.48 | 29.89 ± 6.79 | 65.00 ± 4.32 | 34.33 ± 5.79 | | | |
| pp2048 @ d8192 (c2) | 1240.22 ± 2.98 | 699.57 ± 79.02 | | | 2967.57 ± 334.98 | 2965.36 ± 334.98 | 2967.57 ± 334.98 |
| tg128 @ d8192 (c2) | 54.38 ± 0.36 | 29.63 ± 2.00 | 66.67 ± 0.47 | 33.50 ± 0.50 | | | |
| ctx_pp @ d8192 (c5) | 6354.00 ± 22.23 | 2061.62 ± 811.99 | | | 4540.85 ± 1492.64 | 4538.64 ± 1492.64 | 4540.85 ± 1492.64 |
| ctx_tg @ d8192 (c5) | 60.17 ± 1.85 | 16.80 ± 2.90 | 109.33 ± 2.36 | 23.27 ± 1.48 | | | |
| pp2048 @ d8192 (c5) | 1248.42 ± 1.42 | 418.13 ± 164.31 | | | 5624.15 ± 1913.75 | 5621.94 ± 1913.75 | 5624.15 ± 1913.75 |
| tg128 @ d8192 (c5) | 53.11 ± 1.13 | 15.71 ± 4.09 | 109.00 ± 2.16 | 24.60 ± 2.06 | | | |
| ctx_pp @ d8192 (c10) | 6358.12 ± 3.70 | 1401.80 ± 869.42 | | | 7793.65 ± 3515.30 | 7791.44 ± 3515.30 | 7793.65 ± 3515.30 |
| ctx_tg @ d8192 (c10) | 63.55 ± 4.64 | 10.43 ± 3.36 | 162.33 ± 3.68 | 18.33 ± 1.92 | | | |
| pp2048 @ d8192 (c10) | 1258.45 ± 1.20 | 283.32 ± 177.40 | | | 9727.45 ± 4453.70 | 9725.25 ± 4453.70 | 9727.45 ± 4453.70 |
| tg128 @ d8192 (c10) | 56.53 ± 1.18 | 9.66 ± 3.32 | 163.67 ± 0.47 | 18.30 ± 1.75 | | | |
| ctx_pp @ d16384 (c1) | 5872.30 ± 13.84 | 5872.30 ± 13.84 | | | 2792.45 ± 6.57 | 2790.24 ± 6.57 | 2794.21 ± 6.78 |
| ctx_tg @ d16384 (c1) | 48.87 ± 9.55 | 48.87 ± 9.55 | 57.00 ± 18.38 | 57.00 ± 18.38 | | | |
| pp2048 @ d16384 (c1) | 1115.46 ± 2.40 | 1115.46 ± 2.40 | | | 1838.22 ± 3.96 | 1836.02 ± 3.96 | 1840.65 ± 3.95 |
| tg128 @ d16384 (c1) | 42.42 ± 1.63 | 42.42 ± 1.63 | 44.67 ± 1.25 | 44.67 ± 1.25 | | | |
| ctx_pp @ d16384 (c2) | 6058.92 ± 18.69 | 3628.48 ± 597.13 | | | 4643.46 ± 763.77 | 4641.26 ± 763.77 | 4644.45 ± 764.12 |
| ctx_tg @ d16384 (c2) | 40.47 ± 6.79 | 27.44 ± 3.27 | 62.33 ± 5.91 | 34.45 ± 3.95 | | | |
| pp2048 @ d16384 (c2) | 1179.19 ± 6.96 | 656.19 ± 66.07 | | | 3155.23 ± 317.39 | 3153.02 ± 317.39 | 3156.88 ± 317.26 |
| tg128 @ d16384 (c2) | 55.12 ± 0.66 | 29.98 ± 1.83 | 67.00 ± 0.82 | 34.00 ± 0.82 | | | |
| ctx_pp @ d16384 (c5) | 6165.68 ± 2.26 | 2181.32 ± 982.52 | | | 8931.91 ± 3322.25 | 8929.70 ± 3322.25 | 8933.18 ± 3322.44 |
| ctx_tg @ d16384 (c5) | 37.80 ± 2.27 | 13.51 ± 5.04 | 107.67 ± 2.05 | 24.20 ± 2.26 | | | |
| pp2048 @ d16384 (c5) | 1209.89 ± 1.51 | 392.69 ± 150.03 | | | 5934.35 ± 1941.05 | 5932.14 ± 1941.05 | 5935.70 ± 1941.21 |
| tg128 @ d16384 (c5) | 55.18 ± 0.18 | 15.70 ± 3.46 | 109.33 ± 0.47 | 22.87 ± 0.88 | | | |
| ctx_pp @ d16384 (c10) | 6150.79 ± 1.46 | 1454.82 ± 995.66 | | | 15686.51 ± 7383.00 | 15684.30 ± 7383.00 | 15688.38 ± 7383.01 |
| ctx_tg @ d16384 (c10) | 38.66 ± 1.28 | 8.12 ± 4.19 | 156.00 ± 2.94 | 19.53 ± 3.46 | | | |
| pp2048 @ d16384 (c10) | 1211.69 ± 0.40 | 266.57 ± 162.74 | | | 10205.24 ± 4601.66 | 10203.03 ± 4601.66 | 10207.10 ± 4601.82 |
| tg128 @ d16384 (c10) | 54.09 ± 2.89 | 9.61 ± 3.56 | 156.67 ± 3.86 | 18.40 ± 2.22 | | | |
| ctx_pp @ d32768 (c1) | 5469.26 ± 2.06 | 5469.26 ± 2.06 | | | 5993.70 ± 2.26 | 5991.49 ± 2.26 | 5997.18 ± 2.29 |
| ctx_tg @ d32768 (c1) | 40.85 ± 1.03 | 40.85 ± 1.03 | 43.33 ± 0.47 | 43.33 ± 0.47 | | | |
| pp2048 @ d32768 (c1) | 1031.36 ± 0.29 | 1031.36 ± 0.29 | | | 1987.94 ± 0.55 | 1985.73 ± 0.55 | 1991.71 ± 0.82 |
| tg128 @ d32768 (c1) | 41.80 ± 1.05 | 41.80 ± 1.05 | 44.33 ± 0.47 | 44.33 ± 0.47 | | | |
| ctx_pp @ d32768 (c2) | 5673.76 ± 3.86 | 3376.09 ± 538.01 | | | 9961.33 ± 1587.07 | 9959.13 ± 1587.07 | 9963.81 ± 1587.07 |
| ctx_tg @ d32768 (c2) | 32.51 ± 2.92 | 24.67 ± 6.68 | 65.67 ± 0.94 | 35.00 ± 3.61 | | | |
| pp2048 @ d32768 (c2) | 1080.09 ± 8.34 | 645.23 ± 104.45 | | | 3261.64 ± 527.55 | 3259.43 ± 527.55 | 3265.32 ± 527.67 |
| tg128 @ d32768 (c2) | 47.49 ± 1.99 | 28.75 ± 3.18 | 68.00 ± 1.41 | 35.50 ± 3.40 | | | |
| ctx_pp @ d32768 (c5) | 5750.33 ± 5.64 | 2052.89 ± 928.06 | | | 19004.23 ± 7106.93 | 19002.02 ± 7106.93 | 19007.00 ± 7107.09 |
| ctx_tg @ d32768 (c5) | 21.71 ± 1.58 | 10.69 ± 6.93 | 99.33 ± 5.19 | 23.27 ± 6.31 | | | |
| pp2048 @ d32768 (c5) | 1114.21 ± 8.26 | 361.85 ± 153.58 | | | 6556.05 ± 2215.72 | 6553.84 ± 2215.72 | 6558.05 ± 2215.67 |
| tg128 @ d32768 (c5) | 52.20 ± 1.44 | 15.62 ± 3.81 | 111.00 ± 3.56 | 22.73 ± 0.77 | | | |
| ctx_pp @ d32768 (c10) | 5698.73 ± 20.16 | 1351.54 ± 926.61 | | | 33812.24 ± 15959.73 | 33810.03 ± 15959.73 | 33814.01 ± 15959.84 |
| ctx_tg @ d32768 (c10) | 20.37 ± 1.58 | 6.03 ± 4.55 | 147.67 ± 6.85 | 19.97 ± 6.23 | | | |
| pp2048 @ d32768 (c10) | 1096.61 ± 1.16 | 244.98 ± 163.51 | | | 11327.40 ± 5084.11 | 11325.19 ± 5084.11 | 11328.36 ± 5083.84 |
| tg128 @ d32768 (c10) | 51.54 ± 1.73 | 9.22 ± 3.41 | 159.00 ± 2.83 | 17.77 ± 2.09 | | | |
| ctx_pp @ d65535 (c1) | 4776.14 ± 24.46 | 4776.14 ± 24.46 | | | 13724.11 ± 70.26 | 13721.90 ± 70.26 | 13728.56 ± 70.22 |
| ctx_tg @ d65535 (c1) | 39.60 ± 2.81 | 39.60 ± 2.81 | 42.33 ± 2.05 | 42.33 ± 2.05 | | | |
| pp2048 @ d65535 (c1) | 922.72 ± 3.42 | 922.72 ± 3.42 | | | 2221.76 ± 8.22 | 2219.55 ± 8.22 | 2226.26 ± 7.35 |
| tg128 @ d65535 (c1) | 41.08 ± 0.90 | 41.08 ± 0.90 | 44.67 ± 1.25 | 44.67 ± 1.25 | | | |
| ctx_pp @ d65535 (c2) | 4919.83 ± 8.18 | 2991.51 ± 530.83 | | | 22621.69 ± 4013.58 | 22619.48 ± 4013.58 | 22627.58 ± 4013.91 |
| ctx_tg @ d65535 (c2) | 16.29 ± 6.34 | 21.38 ± 14.02 | 57.67 ± 10.40 | 30.67 ± 12.83 | | | |
| pp2048 @ d65535 (c2) | 955.84 ± 7.61 | 579.86 ± 101.13 | | | 3644.77 ± 634.86 | 3642.56 ± 634.86 | 3649.96 ± 636.18 |
| tg128 @ d65535 (c2) | 47.65 ± 0.45 | 27.75 ± 3.49 | 68.67 ± 2.05 | 34.33 ± 1.11 | | | |
| ctx_pp @ d65535 (c5) | 5036.75 ± 8.68 | 1774.12 ± 782.87 | | | 43586.65 ± 15888.47 | 43584.44 ± 15888.47 | 43589.70 ± 15888.71 |
| ctx_tg @ d65535 (c5) | 10.97 ± 1.16 | 8.43 ± 8.00 | 103.33 ± 2.05 | 26.13 ± 6.95 | | | |
| pp2048 @ d65535 (c5) | 1008.08 ± 0.67 | 352.05 ± 179.72 | | | 7055.81 ± 2564.57 | 7053.60 ± 2564.57 | 7057.04 ± 2564.56 |
| tg128 @ d65535 (c5) | 46.00 ± 0.31 | 14.35 ± 3.56 | 105.00 ± 1.41 | 21.53 ± 0.88 | | | |
| ctx_pp @ d65535 (c10) | 5005.60 ± 7.84 | 1181.48 ± 800.57 | | | 76904.08 ± 36124.00 | 76901.87 ± 36124.00 | 76908.10 ± 36123.87 |
| ctx_tg @ d65535 (c10) | 9.82 ± 0.41 | 4.94 ± 6.68 | 125.00 ± 10.68 | 17.10 ± 12.38 | | | |
| pp2048 @ d65535 (c10) | 1012.81 ± 3.73 | 233.91 ± 171.19 | | | 12253.26 ± 5603.77 | 12251.05 ± 5603.77 | 12256.19 ± 5603.30 |
| tg128 @ d65535 (c10) | 47.54 ± 1.38 | 8.64 ± 3.02 | 156.33 ± 5.31 | 16.90 ± 1.60 | | | |
| ctx_pp @ d131072 (c1) | 3928.76 ± 10.42 | 3928.76 ± 10.42 | | | 33364.86 ± 88.64 | 33362.65 ± 88.64 | 33373.33 ± 88.51 |
| ctx_tg @ d131072 (c1) | 40.93 ± 0.11 | 40.93 ± 0.11 | 46.00 ± 0.82 | 46.00 ± 0.82 | | | |
| pp2048 @ d131072 (c1) | 556.23 ± 1.35 | 556.23 ± 1.35 | | | 3684.16 ± 8.97 | 3681.95 ± 8.97 | 3693.93 ± 7.47 |
| tg128 @ d131072 (c1) | 39.75 ± 1.09 | 39.75 ± 1.09 | 44.67 ± 0.94 | 44.67 ± 0.94 | | | |
| ctx_pp @ d131072 (c2) | 4029.51 ± 1.18 | 2479.14 ± 464.04 | | | 54792.11 ± 10255.31 | 54789.90 ± 10255.31 | 54798.85 ± 10257.07 |
| ctx_tg @ d131072 (c2) | 9.16 ± 1.45 | 18.42 ± 13.09 | 65.67 ± 1.25 | 36.00 ± 3.83 | | | |
| pp2048 @ d131072 (c2) | 581.05 ± 2.80 | 351.35 ± 60.30 | | | 6008.00 ± 1030.59 | 6005.79 ± 1030.59 | 6016.33 ± 1032.72 |
| tg128 @ d131072 (c2) | 36.53 ± 4.25 | 25.33 ± 3.99 | 65.33 ± 0.47 | 34.33 ± 3.99 | | | |
| ctx_pp @ d131072 (c5) | 4171.48 ± 71.62 | 1477.58 ± 662.72 | | | 105169.70 ± 38805.91 | 105167.50 ± 38805.91 | 105176.77 ± 38806.85 |
| ctx_tg @ d131072 (c5) | 4.85 ± 0.67 | 7.89 ± 11.13 | 80.00 ± 10.61 | 20.57 ± 14.26 | | | |
| pp2048 @ d131072 (c5) | 621.50 ± 18.34 | 214.22 ± 100.35 | | | 11378.21 ± 4081.27 | 11376.00 ± 4081.27 | 11381.73 ± 4081.09 |
| tg128 @ d131072 (c5) | 33.99 ± 2.03 | 12.22 ± 4.12 | 105.00 ± 1.41 | 21.40 ± 0.95 | | | |
| ctx_pp @ d131072 (c10) | 4204.59 ± 50.71 | 992.33 ± 680.35 | | | 183688.55 ± 86207.66 | 183686.35 ± 86207.66 | 183693.66 ± 86207.21 |
| ctx_tg @ d131072 (c10) | 4.18 ± 0.37 | 4.70 ± 9.13 | 85.00 ± 13.95 | 11.97 ± 14.56 | | | |
| pp2048 @ d131072 (c10) | 627.11 ± 17.12 | 144.39 ± 99.73 | | | 19569.16 ± 9014.07 | 19566.95 ± 9014.07 | 19571.63 ± 9013.42 |
| tg128 @ d131072 (c10) | 32.39 ± 1.51 | 6.73 ± 3.10 | 141.33 ± 0.47 | 15.93 ± 2.11 | | | |
| ctx_pp @ d262144 (c1) | 2981.79 ± 21.77 | 2981.79 ± 21.77 | | | 87922.23 ± 642.95 | 87920.02 ± 642.95 | 87936.77 ± 643.82 |
| ctx_tg @ d262144 (c1) | 37.12 ± 0.52 | 37.12 ± 0.52 | 44.00 ± 0.82 | 44.00 ± 0.82 | | | |
| pp2048 @ d262144 (c1) | 384.10 ± 0.79 | 384.10 ± 0.79 | | | 5334.16 ± 11.00 | 5331.95 ± 11.00 | 5348.44 ± 10.63 |
| tg128 @ d262144 (c1) | 37.71 ± 0.45 | 37.71 ± 0.45 | 45.33 ± 0.47 | 45.33 ± 0.47 | | | |
| ctx_pp @ d262144 (c2) | 3079.95 ± 16.65 | 1912.67 ± 372.77 | | | 142469.32 ± 27759.96 | 142467.11 ± 27759.96 | 142484.03 ± 27761.94 |
| ctx_tg @ d262144 (c2) | 4.16 ± 0.05 | 16.64 ± 14.46 | 65.33 ± 0.47 | 35.17 ± 3.08 | | | |
| pp2048 @ d262144 (c2) | 408.29 ± 0.22 | 245.19 ± 40.60 | | | 8590.35 ± 1421.98 | 8588.14 ± 1421.98 | 8604.09 ± 1426.73 |
| tg128 @ d262144 (c2) | 29.92 ± 4.88 | 21.71 ± 9.60 | 62.00 ± 2.83 | 31.00 ± 6.51 | | | |
| ctx_pp @ d262144 (c5) | 3122.28 ± 21.61 | 1126.91 ± 518.43 | | | 278075.89 ± 104777.67 | 278073.68 ± 104777.67 | 278087.25 ± 104779.16 |
| ctx_tg @ d262144 (c5) | 1.69 ± 0.30 | 7.16 ± 11.82 | 65.00 ± 0.82 | 15.27 ± 15.69 | | | |
| pp2048 @ d262144 (c5) | 427.79 ± 0.29 | 145.57 ± 63.97 | | | 16455.55 ± 5726.42 | 16453.34 ± 5726.42 | 16466.23 ± 5728.54 |
| tg128 @ d262144 (c5) | 26.23 ± 0.25 | 9.70 ± 3.78 | 99.00 ± 2.94 | 19.87 ± 0.62 | | | |
| ctx_pp @ d262144 (c10) | 3169.13 ± 6.23 | 758.50 ± 525.39 | | | 483606.30 ± 228784.97 | 483604.09 ± 228784.97 | 483614.31 ± 228784.40 |
| ctx_tg @ d262144 (c10) | 1.61 ± 0.13 | 4.39 ± 10.02 | 58.67 ± 10.37 | 8.43 ± 13.44 | | | |
| pp2048 @ d262144 (c10) | 432.99 ± 0.21 | 97.62 ± 64.99 | | | 28533.84 ± 12943.60 | 28531.63 ± 12943.60 | 28539.40 ± 12944.55 |
| tg128 @ d262144 (c10) | 22.47 ± 1.67 | 5.31 ± 3.16 | 125.33 ± 4.71 | 14.33 ± 4.09 | | | |
| ctx_pp @ d524288 (c1) | 2015.62 ± 5.05 | 2015.62 ± 5.05 | | | 260117.23 ± 653.08 | 260115.02 ± 653.08 | 260152.26 ± 660.33 |
| ctx_tg @ d524288 (c1) | 31.95 ± 1.40 | 31.95 ± 1.40 | 46.00 ± 0.82 | 46.00 ± 0.82 | | | |
| pp2048 @ d524288 (c1) | 238.90 ± 0.34 | 238.90 ± 0.34 | | | 8574.79 ± 12.08 | 8572.58 ± 12.08 | 8609.22 ± 15.53 |
| tg128 @ d524288 (c1) | 31.71 ± 1.05 | 31.71 ± 1.05 | 44.01 ± 1.43 | 44.01 ± 1.43 | | | |
| ctx_pp @ d524288 (c2) | 2057.80 ± 3.40 | 1301.44 ± 272.48 | | | 421324.23 ± 88209.65 | 421322.02 ± 88209.65 | 421350.44 ± 88216.70 |
| ctx_tg @ d524288 (c2) | 1.32 ± 0.01 | 16.58 ± 15.86 | 45.67 ± 1.70 | 23.83 ± 21.54 | | | |
| pp2048 @ d524288 (c2) | 255.26 ± 0.44 | 151.47 ± 23.51 | | | 13856.22 ± 2149.79 | 13854.01 ± 2149.79 | 13887.45 ± 2155.66 |
| tg128 @ d524288 (c2) | 23.13 ± 3.55 | 17.92 ± 7.74 | 63.33 ± 2.87 | 32.00 ± 1.63 | | | |
| ctx_pp @ d524288 (c5) | 2091.76 ± 2.99 | 778.92 ± 374.41 | | | 817105.49 ± 319393.11 | 817103.28 ± 319393.11 | 817129.56 ± 319396.79 |
| ctx_tg @ d524288 (c5) | 0.51 ± 0.05 | 7.06 ± 12.63 | 44.67 ± 1.89 | 10.85 ± 16.93 | | | |
| pp2048 @ d524288 (c5) | 272.54 ± 0.40 | 86.25 ± 36.38 | | | 27354.27 ± 8928.51 | 27352.06 ± 8928.51 | 27383.73 ± 8927.39 |
| tg128 @ d524288 (c5) | 13.97 ± 0.94 | 7.90 ± 4.61 | 88.33 ± 3.77 | 20.30 ± 7.40 | | | |
| ctx_pp @ d524288 (c10) | 2106.45 ± 2.43 | 519.39 ± 373.89 | | | 1439815.26 ± 693536.79 | 1439813.05 ± 693536.79 | 1439840.62 ± 693538.18 |
| ctx_tg @ d524288 (c10) | 0.52 ± 0.07 | 2.87 ± 8.16 | 33.67 ± 18.87 | 5.40 ± 11.16 | | | |
| pp2048 @ d524288 (c10) | 43.70 ± 11.99 | 53.45 ± 47.20 | | | 115240.78 ± 160875.93 | 115238.57 ± 160875.93 | 115369.78 ± 160829.66 |
| tg128 @ d524288 (c10) | 1.99 ± 0.32 | 3.74 ± 9.36 | 61.00 ± 22.64 | 7.97 ± 12.38 | | | |

**Key highlights:**
- **Prefill throughput:** ~5,750–6,550 t/s total at concurrency levels 1–10 (strong prefill parallelism)
- **Peak generation (c10):** 177.67 t/s total (at c10) — scales well under load
- **Single session generation:** ~41–44 t/s peak (consistent with DSpark speculative decoding budget)
- **Deep context scaling:** Evaluated across context depths up to 512K (`d524288`). At 512K context (`d524288 (c1)`), prefill throughput remains ~2,015 t/s.



---

### 6. Stop the Server

To stop and remove the server container:

```bash
bash docker/stop.sh
```
