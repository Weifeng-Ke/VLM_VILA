#!/usr/bin/env python3
"""
Benchmark ViLA 1.5-3B with and without prefix caching.

Metrics:
    - TTFT  (time to first token)
    - Total latency
    - Throughput (tokens/sec)
"""

import time
import torch
from termcolor import colored
import llava
from llava import conversation as clib
from llava.media import Image

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
MODEL_PATH   = "Efficient-Large-Model/VILA1.5-3b"
IMAGE_PATH   = "/workspace/VILA/demo_images/demo_img.png"
TEXT_PROMPT  = "Please describe the image."
CONV_MODE    = "vicuna_v1"
MAX_NEW_TOKENS = 128

# ----------------------------------------------------------
# Load model
# ----------------------------------------------------------
print(colored("Loading ViLA model...", "yellow"))
model = llava.load(MODEL_PATH, model_base=None)
clib.default_conversation = clib.conv_templates[CONV_MODE].copy()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------------
# Helper: run one generation and record times
# ----------------------------------------------------------
def timed_generate(model, prompt, use_cache: bool):
    model.enable_prefix_cache = use_cache
    torch.cuda.empty_cache()
    start_time = time.perf_counter()

    # Tokenize + prepare
    input_prep_time = time.perf_counter()

    # Run generation (synchronous)
    with torch.inference_mode():
        torch.cuda.synchronize()
        gen_start = time.perf_counter()
        output = model.generate_content(prompt)
        torch.cuda.synchronize()
        end_time = time.perf_counter()

    total_latency = end_time - start_time
    inference_time = end_time - gen_start
    # Simple proxy for TTFT (ViLA doesn’t stream tokens; we approximate)
    ttft = 0.0

    # Rough token count
    n_tokens = len(output.split())
    throughput = n_tokens / inference_time if inference_time > 0 else float("nan")

    return {
        "output": output,
        "total_latency": total_latency,
        "inference_time": inference_time,
        "ttft": ttft,
        "throughput": throughput,
    }

# ----------------------------------------------------------
# Prepare media prompt
# ----------------------------------------------------------
img = Image(IMAGE_PATH)
prompt = [img, TEXT_PROMPT]

# ----------------------------------------------------------
# Run benchmarks
# ----------------------------------------------------------
print(colored("Benchmarking WITHOUT prefix cache...", "red"))
no_cache_result = timed_generate(model, prompt, use_cache=False)

print(colored("Benchmarking WITH prefix cache...", "green"))
cache_result = timed_generate(model, prompt, use_cache=True)

# ----------------------------------------------------------
# Display results
# ----------------------------------------------------------
def fmt(sec):
    return f"{sec*1000:.1f} ms"

print("\n" + "="*60)
print(colored("RESULTS", "cyan", attrs=["bold"]))
print("="*60)
print(f"Total latency (no cache):   {fmt(no_cache_result['total_latency'])}")
print(f"Total latency (with cache): {fmt(cache_result['total_latency'])}")
print(f"Inference (decoder) time (no cache):   {fmt(no_cache_result['inference_time'])}")
print(f"Inference (decoder) time (with cache): {fmt(cache_result['inference_time'])}")
print(f"Throughput (no cache):   {no_cache_result['throughput']:.2f} tokens/sec")
print(f"Throughput (with cache): {cache_result['throughput']:.2f} tokens/sec")
print("="*60)

print(colored("\nExample output (with cache):", "yellow"))
print(colored(cache_result["output"], "cyan", attrs=["bold"]))
