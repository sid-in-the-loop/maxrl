#!/bin/bash
#SBATCH --job-name=polaris@1e5
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-gpu=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/ssmurali/maxrl/logs/idk/polaris_1e5_%j.out
#SBATCH --error=/home/ssmurali/maxrl/logs/idk/polaris_1e5_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

checkpoint() { echo "[$(date)] Preemption signal received"; }
trap checkpoint USR1

# ---------- Ray tmpdir (per-job isolation) ----------
MACHINE_SPECIFIC_RAY_DIR="/tmp/ray_$(hostname)_$(whoami)_$$"
mkdir -p $MACHINE_SPECIFIC_RAY_DIR
export RAY_TMPDIR=$MACHINE_SPECIFIC_RAY_DIR

# ---------- conda ----------
source /data/user_data/ssmurali/miniconda3/etc/profile.d/conda.sh
conda activate maxrl

CONDA_LIB=/data/user_data/ssmurali/miniconda3/envs/maxrl/lib
NVIDIA_LIBS=$(python -c "import site; pkgs=site.getsitepackages()[0]; import glob,os; dirs=[os.path.dirname(p) for p in glob.glob(pkgs+'/nvidia/*/lib/')]; print(':'.join(dirs))" 2>/dev/null)
export LD_LIBRARY_PATH=${CONDA_LIB}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}

# ---------- env vars ----------
export PYTHONUNBUFFERED=1
export RAY_BACKEND_LOG_LEVEL=warning
export RAY_LOG_TO_STDERR=0
export WANDB_MODE=online
export WANDB_START_METHOD=thread
export WANDB_PROJECT=idk-abstention
export HYDRA_FULL_ERROR=1
export NCCL_P2P_DISABLE=1
export VLLM_USE_V1=1
export RAY_raylet_start_wait_time_s=30
export WANDB_API_KEY=${WANDB_API_KEY}
export HF_TOKEN=${HF_TOKEN}
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}

# ---------- paths ----------
SAVE_DIR=/data/user_data/ssmurali/idk/polaris_1e5_run
REWARD_FN=/home/ssmurali/maxrl/idk/reward_fn.py
TRAIN_DATA=/home/ssmurali/maxrl/data/polaris_idk/train.parquet

python -c "import os; os.makedirs('$SAVE_DIR', exist_ok=True)"
mkdir -p /home/ssmurali/maxrl/logs/idk/rollouts

cd /home/ssmurali/maxrl

echo "[$(date)] IDK GRPO — Polaris, lr=1e-5, bs=32, 2×L40S"

# Kill stale Ray processes from previous jobs
ray stop --force 2>/dev/null || true
sleep 2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    data.train_files=$TRAIN_DATA \
    data.val_files=null \
    data.prompt_key=prompt \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    data.truncation=left \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    "+actor_rollout_ref.model.override_config={attn_implementation: sdpa}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.grad_clip=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=5120 \
    actor_rollout_ref.rollout.disable_log_stats=True \
    +actor_rollout_ref.rollout.logprobs_mode=null \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    custom_reward_function.path=$REWARD_FN \
    custom_reward_function.name=compute_score \
    reward_model.enable=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.total_training_steps=100 \
    trainer.total_epochs=99999 \
    trainer.save_freq=25 \
    trainer.test_freq=0 \
    trainer.val_before_train=False \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb']" \
    trainer.project_name=idk-abstention \
    trainer.experiment_name=polaris_1e5 \
    trainer.default_local_dir=$SAVE_DIR \
    ray_init.ray_dir=$MACHINE_SPECIFIC_RAY_DIR \
    trainer.resume_mode=auto \
    trainer.rollout_data_dir=/home/ssmurali/maxrl/logs/idk/rollouts/polaris_1e5

echo "[$(date)] Training completed."
