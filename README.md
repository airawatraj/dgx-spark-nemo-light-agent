# DGX Spark - Nemotron 3.5 Lightning Serving & Benchmarking

This repository contains the minimal code structure and scripts to serve and benchmark the **NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4** model (using speculative decoding with **NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark**) on a DGX Spark.

The model is served under the name `Cogni-Brain` inside a Docker container named `spark-brain`.

## Quick Start

### 1. Environment Setup

Run the installation script to disable OS swap, verify Docker, and install the `uv` tool (used for managing python environments and running scripts):

```bash
bash setup/install.sh
```

### 2. Download Model Weights

Download the main model weights and the speculative DSpark weights from the Hugging Face hub. If you are downloading a gated model or need authentication, export your Hugging Face token first:

```bash
export HF_TOKEN="your_huggingface_api_token"
bash setup/download_model.sh
```

By default, weights are cached at `$HOME/.cache/huggingface`.

### 3. Start the Server

Launch the vLLM container on DGX Spark. This mounts your local Hugging Face cache and runs vLLM with the optimized speculative decoding setup, 1M context configuration, and appropriate reasoning and tool-choice parsers:

```bash
export HF_TOKEN="your_huggingface_api_token"  # Required if model requires HF auth
bash docker/start.sh
```

To monitor startup progress:
```bash
docker logs -f spark-brain
```

### 4. Check Status

Verify the health, memory usage, and KV cache stats of the container:

```bash
bash docker/status.sh
```

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

---

### 6. Stop the Server

To stop and remove the server container:

```bash
bash docker/stop.sh
```
