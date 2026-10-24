#!/usr/bin/env bash
# Download the evaluation benchmarks into data/raw/.
# MATH-500 comes from OpenAI's PRM800K split and is checksum-verified; the others are the
# copies used by the Qwen2.5-Math evaluation harness. AIME 2025 is fetched from Hugging Face.
set -euo pipefail

MATH500_SHA256=35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132
QWEN_DATA=https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data
AIME25_URL=https://huggingface.co/datasets/math-ai/aime25/resolve/563bb8404243c5f09de6ec262f2db674fe5bce9b/test.jsonl
AIME25_SHA256=b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331

cd "$(dirname "$0")"
mkdir -p raw
cd raw

curl -sSfL -o math500.jsonl \
  https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits/test.jsonl
if command -v sha256sum >/dev/null; then
  echo "$MATH500_SHA256  math500.jsonl" | sha256sum -c -
else
  echo "$MATH500_SHA256  math500.jsonl" | shasum -a 256 -c -
fi

for name in aime24 amc23 minerva_math olympiadbench gsm8k; do
  curl -sSfL -o "$name.jsonl" "$QWEN_DATA/$name/test.jsonl"
done

curl -sSfL -o aime25.jsonl "$AIME25_URL"
if command -v sha256sum >/dev/null; then
  echo "$AIME25_SHA256  aime25.jsonl" | sha256sum -c -
else
  echo "$AIME25_SHA256  aime25.jsonl" | shasum -a 256 -c -
fi

wc -l ./*.jsonl
