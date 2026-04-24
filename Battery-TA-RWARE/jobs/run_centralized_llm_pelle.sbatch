#!/bin/bash
#SBATCH -A YOUR_PROJECT_CODE
#SBATCH -J central_llm
#SBATCH -p pelle
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 10:00:00
#SBATCH --mem=32G
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Keep Python/numerical libraries inside the CPUs requested above.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export VECLIB_MAXIMUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Keep Ollama data somewhere predictable. Change this if your models live elsewhere.
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"

# If you use modules/conda/venv on Pelle, load or activate it here, for example:
# module load python/3.11
# source "$HOME/path/to/venv/bin/activate"

cd Battery-TA-RWARE

# Start the Ollama server inside the Slurm allocation.
ollama serve > "../logs/ollama-${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null

python scripts/run_centralized_llm_experiments.py \
  --ollama_url "http://${OLLAMA_HOST}/api/generate" \
  --results_dir results/central_llm_experiments
