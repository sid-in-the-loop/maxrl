#!/bin/bash
#SBATCH --job-name=idk_eval
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ssmurali/maxrl/logs/idk/eval_%j.out
#SBATCH --error=/home/ssmurali/maxrl/logs/idk/eval_%j.err

source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate verl

CONDA_LIB=/data/user_data/ssmurali/miniconda3/envs/verl/lib
NVIDIA_LIBS=$(python -c "import site; pkgs=site.getsitepackages()[0]; import glob,os; dirs=[os.path.dirname(p) for p in glob.glob(pkgs+'/nvidia/*/lib/')]; print(':'.join(dirs))" 2>/dev/null)
export LD_LIBRARY_PATH=${CONDA_LIB}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1

cd /home/ssmurali/maxrl

EVAL_FILES="/home/ssmurali/maxrl/data/math12k_val.json /home/ssmurali/maxrl/data/polaris_val.json"
OUTPUT="idk/eval_results.jsonl"

# Clear previous results
rm -f $OUTPUT

echo "[$(date)] Evaluating Polaris model checkpoints..."
python3 idk/eval_checkpoints.py \
    --ckpt_dir /data/user_data/ssmurali/idk/job_1 \
    --eval_files $EVAL_FILES \
    --model_name polaris_model \
    --output $OUTPUT \
    --n_samples 200

echo "[$(date)] Evaluating MATH-12k model checkpoints..."
python3 idk/eval_checkpoints.py \
    --ckpt_dir /data/user_data/ssmurali/idk/job_3 \
    --eval_files $EVAL_FILES \
    --model_name math12k_model \
    --output $OUTPUT \
    --n_samples 200

echo "[$(date)] Generating plots..."
python3 idk/plot_eval.py --input $OUTPUT --output idk/plots/

echo "[$(date)] Done. Plots saved to idk/plots/"
