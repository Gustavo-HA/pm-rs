#!/bin/bash
#SBATCH --partition=C0
#SBATCH --job-name=recsys_run_models
#SBATCH --output=scripts/run_models.log
#SBATCH --error=scripts/run_models.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --time=03:00:00

# Enviar correo electrónico cuando el trabajo finalice o falle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx

# add uv to path
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "=== Inicio: $(date) ==="

echo "Nodo: $SLURMD_NODENAME"
echo "CPUs asignados: $SLURM_CPUS_PER_TASK"

uv run python scripts/run_models.py --k 5 10 20 --output scripts/run_models_results.csv

echo "=== Fin: $(date) ==="
