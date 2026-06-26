#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 '<dem-glob>' <work-dir> <archive-out>" >&2
  exit 64
fi

DEM_GLOB="$1"
WORK_DIR="$2"
ARCHIVE_OUT="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET="${WORK_DIR}/anunaki_dataset_512.npz"
CONFIG="${WORK_DIR}/anunaki_production.yaml"
RUN_DIR="${WORK_DIR}/run"

mkdir -p "${WORK_DIR}" "${RUN_DIR}" "${ARCHIVE_OUT}"

python "${SCRIPT_DIR}/prepare_anunaki_dataset.py" \
  --input-glob "${DEM_GLOB}" \
  --out "${DATASET}" \
  --crop-size 512 \
  --stride 384 \
  --max-crops "${ANUNAKI_MAX_CROPS:-8192}" \
  --seed "${ANUNAKI_SEED:-5819}"

python - <<PY
from pathlib import Path
template = Path("${SCRIPT_DIR}/configs/anunaki_production_apple.yaml").read_text()
template = template.replace("/Volumes/AnunakiData/anunaki_dataset_512.npz", "${DATASET}")
template = template.replace("/Volumes/AnunakiRuns/anunaki_production_mps", "${RUN_DIR}")
Path("${CONFIG}").write_text(template)
PY

python "${SCRIPT_DIR}/train_anunaki_diffusion.py" --config "${CONFIG}" ${ANUNAKI_RESUME:+--resume}

python "${SCRIPT_DIR}/export_anunaki_to_metal.py" \
  --checkpoints "${RUN_DIR}/checkpoints" \
  --out "${ARCHIVE_OUT}" \
  --validate \
  --min-variance "${ANUNAKI_MIN_VARIANCE:-0.002}"

echo "Production Anunaki archives are ready at ${ARCHIVE_OUT}"
