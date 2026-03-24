#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=RecSysRunModels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output="./run_models.log"
#SBATCH --error="./run_models.err"
#SBATCH --mem=0
#SBATCH --time=0

# Enviar correo electrónico cuando el trabajo finalice o falle
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gustavo.angeles@cimat.mx

####################### CONFIGURACION DE UV #######################

# add uv to path
export PATH="/home/est_posgrado_gustavo.angeles/.local/uv/bin:$PATH"


# change to the directory where the job was submitted
cd "/home/est_posgrado_gustavo.angeles/Proyectos/pm-rs/"

####################### EJECUCION DEL SCRIPT DE PYTHON #######################

PYTHON_SCRIPT="./scripts/run_models.py"

echo "Ejecutando el script de Python: $PYTHON_SCRIPT"

# Usar 'uv run' para asegurar que se usen las dependencias del proyecto
# Usar torchrun para DDP
uv run "$PYTHON_SCRIPT" --k 5 10 20 --output "./scripts/results_run_models.csv"

if [ $? -ne 0 ]; then
    echo "ERROR: El script de Python falló."
    exit 1
fi

echo "--- Trabajo de Slurm Finalizado ---"
