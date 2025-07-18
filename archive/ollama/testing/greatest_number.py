#!/usr/bin/env python3
"""
Call a local Ollama LLM to return the larger of two numbers
and save the reply in output.txt.
Usage: python largest_number.py 42 73
"""

import sys, re, ollama, pathlib

if len(sys.argv) != 3:
    sys.exit("Usage: python largest_number.py <num1> <num2>")

a, b = sys.argv[1], sys.argv[2]
prompt = f"Return ONLY the larger of these two numbers: {a} {b}"

resp = ollama.chat(
    model="mistral:7b",              # any model you pulled
    messages=[
        {"role": "system",
         "content": "You are a calculator. Reply with just one number."},
        {"role": "user", "content": prompt},
    ],
    stream=False                     # single JSON result
)

answer = resp["message"]["content"].strip()

# Optional sanity-check: keep the first integer we see
match = re.search(r"-?\d+(?:\.\d+)?", answer)
clean  = match.group(0) if match else answer

pathlib.Path("output.txt").write_text(clean + "\n")
print(f"largest number → {clean}")
