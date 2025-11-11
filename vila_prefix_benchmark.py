#!/usr/bin/env python3
"""
Benchmark ViLA 1.5-3B with and without prefix caching.

Metrics:
    - TTFT  (time to first token, approximated)
    - Total latency
    - Throughput (tokens/sec)

Usage:
    python vila_prefix_benchmark.py --num-runs 5 --max-new-tokens 128
"""

import argparse
import time
import torch
from termcolor import colored
import llava
from llava import conversation as clib
from llava.media import Image

# ----------------------------------------------------------
# Argument Parser
# ----------------------------------------------------------
parser = argparse.ArgumentParser(description="Benchmark ViLA prefix caching.")
parser.add_argument("--num-runs", type=int, default=3, help="Number of repeated runs for averaging.")
parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b", help="Model name or path.")
parser.add_argument("--image-path", type=str, default="/workspace/VILA/demo_images/demo_img.png", help="Path to image file.")
parser.add_argument("--text-prompt", type=str, default="Please describe the image.", help="Text prompt to run.")
parser.add_argument("--conv-mode", type=str, default="vicuna_v1", help="Conversation template mode.")
args = parser.parse_args()

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
MODEL_PATH   = args.model_path
IMAGE_PATH   = args.image_path
TEXT_PROMPT  = args.text_prompt
CONV_MODE    = args.conv_mode
NUM_RUNS     = args.num_runs
MAX_NEW_TOKENS = 128

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
print(colored(f"Loading ViLA model from {MODEL_PATH} ...", "yellow"))
model = llava.load(MODEL_PATH, model_base=None)
clib.default_conversation = clib.conv_templates[CONV_MODE].copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Prepare media prompt
# ----------------------------------------------------------
media = Image(IMAGE_PATH)
prompt = [media, TEXT_PROMPT]

# ----------------------------------------------------------
# Helper: run one generation and record times
# ----------------------------------------------------------
def timed_generate(model, prompt, use_cache: bool):
    model.enable_prefix_cache = use_cache
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate_content(prompt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    total_latency = end - start
    num_tokens = len(output.split())
    throughput = num_tokens / total_latency if total_latency > 0 else float("nan")
    return output, total_latency, throughput

def fmt_ms(sec: float) -> str:
    return f"{sec :.3f} s"

# ----------------------------------------------------------
# Run multiple iterations
# ----------------------------------------------------------
def run_benchmark(use_cache: bool, runs: int):
    times, throughputs = [], []
    for i in range(runs):
        print(colored(f"  Run {i+1}/{runs} (use_cache={use_cache})...", "yellow"))
        _, t, thr = timed_generate(model, prompt, use_cache=use_cache)
        times.append(t)
        throughputs.append(thr)
        print(f"    ↳ latency: {fmt_ms(t):>10} | throughput: {thr:6.2f} tokens/sec")
    return sum(times)/len(times), sum(throughputs)/len(throughputs)

# ----------------------------------------------------------
# Execute benchmarks
# ----------------------------------------------------------
print(colored(f"\nBenchmarking WITHOUT prefix cache ({NUM_RUNS} runs)...", "red"))
avg_t_no_cache, avg_thr_no_cache = run_benchmark(use_cache=False, runs=NUM_RUNS)

# Clear cache between modes to be fair
model.clear_prefix_cache()

print(colored(f"Benchmarking WITH prefix cache ({NUM_RUNS} runs)...", "green"))
avg_t_cache, avg_thr_cache = run_benchmark(use_cache=True, runs=NUM_RUNS)

# Clear cache between modes to be fair
model.clear_prefix_cache()
# ----------------------------------------------------------
# Results
# ----------------------------------------------------------
print("\n" + "=" * 60)
print(colored("BENCHMARK RESULTS", "cyan", attrs=["bold"]))
print("=" * 60)
print(f"Avg latency (no cache):   {fmt_ms(avg_t_no_cache)}")
print(f"Avg latency (with cache): {fmt_ms(avg_t_cache)}")
print(f"Avg throughput (no cache):   {avg_thr_no_cache:.2f} tokens/sec")
print(f"Avg throughput (with cache): {avg_thr_cache:.2f} tokens/sec")
print("=" * 60)
