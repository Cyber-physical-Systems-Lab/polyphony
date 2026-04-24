#!/bin/bash
#SBATCH -A uppmax2026-1-142
#SBATCH -J central_llm
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 10:00:00
#SBATCH --mem=32G
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs ollama_models

IMAGE="${SLURM_SUBMIT_DIR}/Battery-TA-RWARE/jobs/central_llm_ollama.sif"
if [[ ! -f "$IMAGE" ]]; then
  echo "Missing Singularity image: $IMAGE" >&2
  echo "Build it first with:" >&2
  echo "  singularity build Battery-TA-RWARE/jobs/central_llm_ollama.sif Battery-TA-RWARE/jobs/central_llm_ollama.def" >&2
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MODELS="/work/ollama_models"
export PYTHONPATH="/work/Battery-TA-RWARE:${PYTHONPATH:-}"

SINGULARITY_BIN="$(command -v apptainer || command -v singularity)"
SINGULARITY_RUN=(
  "$SINGULARITY_BIN" exec
  --nv
  --bind "${SLURM_SUBMIT_DIR}:/work"
  --pwd /work/Battery-TA-RWARE
  "$IMAGE"
)

"${SINGULARITY_RUN[@]}" nvidia-smi || true

"${SINGULARITY_RUN[@]}" ollama serve > "logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if "${SINGULARITY_RUN[@]}" curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

"${SINGULARITY_RUN[@]}" curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null

"${SINGULARITY_RUN[@]}" python3 scripts/run_obj2_centralized_llm.py \
  --prompt_format json \
  --max_steps 0 \
  --max_busy_steps_for_replan 10 \
  --max_inactivity_steps 500 \
  --disable_support_needed_soon \
  --find_path_agent_aware_always \
  --seed 0 \
  --ollama_url "http://${OLLAMA_HOST}/api/generate" \
  --results_dir results/calls/centralized/json
