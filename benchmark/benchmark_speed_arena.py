#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "httpx",
# ]
# ///
"""
DGX Spark / Nemotron-3.5-Lightning-30B Full Benchmark (spark-arena style)
Runs the long-form llama-benchy sweep used for spark-arena-style measurements.
Automatically queries the vLLM server to bound depths safely within max_model_len.

Usage:
  uv run benchmark/benchmark_speed_arena.py
  uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import httpx


COLORS = {
    "green":  "\033[92m",
    "yellow": "\033[93m",
    "red":    "\033[91m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
    "dim":    "\033[2m",
}


def c(text, color):
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def header(title):
    line = "─" * 60
    print(f"\n{c(line, 'cyan')}")
    print(f"{c('  ' + title, 'bold')}")
    print(f"{c(line, 'cyan')}")


def result_line(label, value, color="green"):
    print(f"  {c(label.ljust(30), 'dim')} {c(str(value), color)}")


def probe_server_max_model_len(base_url: str) -> int:
    """Probe the vLLM server to discover configured max_model_len."""
    # Strip /v1 if present for root endpoints
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[:-3]

    try:
        with httpx.Client(timeout=5.0) as client:
            # 1. Try /load or /metrics or /v1/models
            resp = client.get(f"{root_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("data", []):
                    if "max_model_len" in m:
                        return int(m["max_model_len"])
    except Exception:
        pass

    return 262144  # Safe default if unreachable


def compute_safe_depths(max_model_len: int, tg: int, custom_depths: list[int] | None = None) -> list[str]:
    """Generate safe depth sweep points that reserve space for tg output tokens."""
    if custom_depths:
        raw_depths = custom_depths
    else:
        # Standard exponential hierarchy
        raw_depths = [0, 4096, 8192, 16384, 32768, 65535, 131072, 262144, 524288, 1048576]

    safe_depths = []
    # Reserve at least tg + 2048 tokens for prompt template, system tokens, and generated tokens
    max_safe_input = max(0, max_model_len - tg - 2048)

    for d in raw_depths:
        if d == 0:
            safe_depths.append("0")
        elif d < max_safe_input:
            safe_depths.append(str(d))
        elif d >= max_safe_input:
            # Round down to nearest 1000 for clean reporting
            clamped = (max_safe_input // 1000) * 1000
            if not safe_depths or int(safe_depths[-1]) < (clamped - 1000):
                safe_depths.append(str(clamped))
            break

    return safe_depths


def build_command(args, safe_depths: list[str]):
    cmd = [
        "uv",
        "tool",
        "run",
        "--from",
        "llama-benchy",
        "llama-benchy",
        "--base-url",
        args.base_url,
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer,
        "--depth",
        *safe_depths,
        "--pp",
        str(args.pp),
        "--tg",
        str(args.tg),
        "--enable-prefix-caching",
        "--concurrency",
        *args.concurrency,
        "--save-result",
        args.save_result,
    ]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run the full overnight llama-benchy sweep against the local Cogni-Brain vLLM endpoint.",
        epilog="Example: uv run benchmark/benchmark_speed_arena.py --save-result benchmark/results_full.csv",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument(
        "--model",
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    )
    parser.add_argument("--served-model-name", default="Cogni-Brain")
    parser.add_argument(
        "--tokenizer",
        default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    )
    parser.add_argument("--pp", type=int, default=2048)
    parser.add_argument("--tg", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=None, help="Explicit max context ceiling (auto-detected if omitted)")
    parser.add_argument("--depth", nargs="+", type=int, default=None, help="Explicit depth list to test")
    parser.add_argument("--concurrency", nargs="+", default=["1", "2", "5", "10"], help="Concurrency levels to sweep")
    parser.add_argument("--save-result", default="results_full.csv")
    args = parser.parse_args()

    header("FULL BENCHMARK — NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 + DSpark (Cogni-Brain)")
    result_line("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result_line("Base URL", args.base_url)
    result_line("Model", args.model)
    result_line("Served model name", args.served_model_name)
    result_line("Tokenizer", args.tokenizer)
    result_line("Output CSV", args.save_result)

    # Detect or use configured max_model_len
    detected_len = args.max_model_len or probe_server_max_model_len(args.base_url)
    safe_depths = compute_safe_depths(detected_len, args.tg, args.depth)

    result_line("Detected Server Context", f"{detected_len:,} tokens")
    result_line("Safe Sweep Depths", ", ".join(safe_depths))
    result_line("Concurrency Levels", ", ".join(args.concurrency))
    print()
    print(f"  {c('All depth points are bounded within (max_model_len - tg) to guarantee 0 HTTP 400 errors.', 'green')}")

    if shutil.which("uv") is None:
        print(f"\n{c('✗ uv is not installed or not on PATH', 'red')}")
        sys.exit(1)

    output_path = Path(args.save_result)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(args, safe_depths)

    header("COMMAND")
    print("  " + " \\\n    ".join(command))

    header("RUNNING")
    print(f"  {c('Streaming llama-benchy output below...', 'dim')}")

    try:
        completed = subprocess.run(command, check=False)
    except KeyboardInterrupt:
        print(f"\n{c('Benchmark interrupted by user', 'yellow')}")
        sys.exit(130)
    except FileNotFoundError:
        print(f"\n{c('✗ Failed to start uv', 'red')}")
        sys.exit(1)

    if completed.returncode == 0:
        header("DONE")
        result_line("Status", "Success")
        result_line("Saved results", args.save_result)
    else:
        header("FAILED")
        result_line("Exit code", completed.returncode, color="red")
        print(f"  {c('llama-benchy did not complete successfully.', 'red')}")
        sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
