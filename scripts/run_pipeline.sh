#!/usr/bin/env bash
#
# The full pipeline in one reproducible command.
#
#   scripts/run_pipeline.sh                 # small smoke run
#   COUNT=5000 EPOCHS=30 scripts/run_pipeline.sh
#
# Every stage is skippable; every artefact lands in a directory you can inspect.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-12345}"
SPLIT_SEED="${SPLIT_SEED:-7}"
CARDS="${CARDS:-40}"
COUNT="${COUNT:-200}"
EPOCHS="${EPOCHS:-10}"
PROFILE="${PROFILE:-realistic}"
RENDER_SCALE="${RENDER_SCALE:-1.0}"
VERSION="${VERSION:-0.1.0}"

SOURCE="datasets/source/pipeline"
SYNTHETIC="datasets/synthetic/pipeline"
EXPERIMENT="experiments/pipeline"

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

step "0. environment"
"$PYTHON" -m cardvault_vision doctor

step "1. bootstrap source cards ($CARDS)"
rm -rf "$SOURCE"
"$PYTHON" -m cardvault_vision dataset bootstrap \
    --out "$SOURCE" --count "$CARDS" --seed "$SEED"

step "2. generate synthetic captures ($COUNT @ $PROFILE, scale $RENDER_SCALE)"
rm -rf "$SYNTHETIC"
"$PYTHON" -m cardvault_vision synthetic generate \
    --source "$SOURCE" --out "$SYNTHETIC" \
    --profile "$PROFILE" --count "$COUNT" --seed "$SEED" \
    --render-scale "$RENDER_SCALE"

step "3. validate the generated dataset"
"$PYTHON" -m cardvault_vision dataset validate --manifest "$SYNTHETIC" --hashes

step "4. card-level split (seed $SPLIT_SEED)"
"$PYTHON" -m cardvault_vision dataset split \
    --manifest "$SYNTHETIC" --seed "$SPLIT_SEED"

step "5. train name OCR ($EPOCHS epochs)"
rm -rf "$EXPERIMENT"
"$PYTHON" -m cardvault_vision train name-ocr \
    --dataset "$SYNTHETIC" --config train-ocr \
    --set "epochs=$EPOCHS" --seed "$SEED" --out "$EXPERIMENT/name-ocr"

step "6. evaluate on the held-out split"
"$PYTHON" -m cardvault_vision test name-ocr \
    --checkpoint "$EXPERIMENT/name-ocr/best.pt" \
    --dataset "$SYNTHETIC" --split test \
    --output "$EXPERIMENT/name-ocr/test-report.json" \
    --predictions "$EXPERIMENT/name-ocr/predictions.jsonl"

step "7. mine hard examples"
"$PYTHON" -m cardvault_vision evaluate hard-examples \
    --predictions "$EXPERIMENT/name-ocr/predictions.jsonl" --top 20

step "8. benchmark"
"$PYTHON" -m cardvault_vision benchmark \
    --checkpoint "$EXPERIMENT/name-ocr/best.pt" \
    --devices cpu --output "$EXPERIMENT/benchmark.json"

step "9. export, validate, register, promote"
"$PYTHON" -m cardvault_vision export export \
    --checkpoint "$EXPERIMENT/name-ocr/best.pt" --version "$VERSION"
"$PYTHON" -m cardvault_vision export validate --path "exports/name-ocr/$VERSION"
"$PYTHON" -m cardvault_vision export register --path "exports/name-ocr/$VERSION"
"$PYTHON" -m cardvault_vision export promote --task name-ocr --version "$VERSION"

printf '\n\033[1;32mPipeline complete.\033[0m\n'
printf '  dataset    : %s\n' "$SYNTHETIC"
printf '  experiment : %s\n' "$EXPERIMENT"
printf '  artifact   : exports/name-ocr/%s\n' "$VERSION"
