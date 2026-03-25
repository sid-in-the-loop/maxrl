# IDK Abstention Experiment — Setup & Training Guide

Train Qwen3-1.7B-Base with GRPO to learn selective abstention (`\boxed{idk}`) on math problems using verl.

## Prerequisites

- 2x NVIDIA L40S GPUs (48GB each)
- SLURM cluster access
- Miniconda at `/data/user_data/ssmurali/miniconda3`

## 1. Environment Setup

```bash
conda activate maxrl
```

## 2. Data Preparation

```bash
cd /home/ssmurali/maxrl
python idk/preprocess.py --dataset polaris --idk   # → data/polaris_idk/train.parquet
python idk/preprocess.py --dataset math12k --idk   # → data/math12k_idk/train.parquet
```

`--idk` appends: *"If you feel even slightly uncertain about the answer, put `\boxed{idk}`."*
Omit `--idk` for standard math prompts (no abstention option).

## 3. Reward Function

Located at `idk/reward_fn.py`. Uses `math_verify` (with exact-match fallback) — no LLM judge needed.

| Response | Score |
|----------|-------|
| Correct answer in `\boxed{}` | **1.0** |
| `\boxed{idk}` | **0.5** |
| Wrong / no `\boxed{}` | **0.0** |

Extra metrics logged to wandb per step:
- `reward_extra/is_idk/mean` — idk rate
- `reward_extra/is_correct/mean` — correct rate
- `group/frac_with_gradient_signal` — fraction of prompts with non-trivial advantage

## 4. Active Job Scripts

| Script | Dataset | LR | WandB name |
|--------|---------|-----|------------|
| `idk/run_polaris_5e6.sh` | Polaris-53K | 5e-6 | `polaris_5e6` |
| `idk/run_polaris_1e5.sh` | Polaris-53K | 1e-5 | `polaris_1e5` |
| `idk/run_math_5e6.sh` | MATH-12K | 5e-6 | `math_5e6` |

### Key Training Parameters

```
Algorithm:              GRPO (no critic)
Model:                  Qwen/Qwen3-1.7B-Base
GPUs:                   2x L40S
TP:                     1
Rollouts per prompt:    n=16
Train batch size:       32 prompts -> 512 sequences/step
Max prompt length:      1024 tokens
Max response length:    4096 tokens
Max model length:       5120 tokens
Training steps:         100
GPU memory util:        0.6
Gradient checkpointing: enabled
Ref model offload:      param_offload=True
Norm adv by std:        True (GRPO default)
```

### Submit

```bash
sbatch idk/run_polaris_5e6.sh
sbatch idk/run_polaris_1e5.sh
sbatch idk/run_math_5e6.sh
```

### Monitor

```bash
squeue -u $USER

# Logs (gitignored)
tail -f logs/idk/<experiment>_<JOBID>.err

# Rollout outputs (JSONL, one file per step)
cat logs/idk/rollouts/polaris_5e6/1.jsonl | head -1 | python3 -m json.tool
```

WandB project: `idk-abstention` at https://wandb.ai/ssmurali-cmu/idk-abstention

## 5. Expected Dynamics

**math12k** (easy dataset): mean reward rises steadily; idk rate stays near 0 — model gets correct answers and has no incentive to abstain.

**polaris** (hard dataset): idk rate climbs as the model learns `\boxed{idk}` gives a consistent +0.5 reward on problems it can't solve. Mean reward converges toward 0.5. This is the reward hacking we're studying.

## 6. Important Lessons

### Base Model + Chat Format
- Use `Qwen3-1.7B-Base` with `data.return_raw_chat=True` and chat-formatted prompts.
- The base model accepts chat templates fine; just don't use the instruct system prompt.

### Memory (L40S 48GB)
- `gpu_memory_utilization=0.6` — safe ceiling
- `enforce_eager=True` — avoids CUDA graph peak memory
- `free_cache_engine=True` — releases vLLM KV cache between rollout and training
- `param_offload=False` for actor (too slow), `True` for ref model

### GRPO Signal
- With `n=16`, if all 16 rollouts get the same score, advantage = 0 -> no gradient.
- `group/frac_with_gradient_signal` tracks what fraction of prompts actually contribute.
- Temperature=1.0 is required for sufficient rollout diversity.

### vllm_rollout Bug (fixed)
- `raw_prompt` (from `return_raw_chat=True`) must be repeated `n` times to match the expanded batch after vLLM generates n rollouts per prompt. Fixed in `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` by looping over all `non_tensor_batch` keys.

### SLURM
- `--mem-per-gpu=64G` required
- Logs go to `logs/idk/` (gitignored)
- Ray tmpdir is per-job: `/tmp/ray_$(hostname)_$(whoami)_$$`

## 7. Directory Structure

```
maxrl/
├── idk/
│   ├── preprocess.py          # Unified data preprocessing (--dataset, --idk flags)
│   ├── reward_fn.py           # Custom reward: correct=1.0, idk=0.5, wrong=0.0
│   ├── run_polaris_5e6.sh     # SLURM job scripts
│   ├── run_polaris_1e5.sh
│   ├── run_math_5e6.sh
│   ├── eval_checkpoints.py    # Evaluation
│   └── guide.md               # This file
├── data/
│   ├── polaris_idk/train.parquet
│   └── math12k_idk/train.parquet
└── logs/idk/                  # gitignored
    ├── <exp>_<JOBID>.out
    ├── <exp>_<JOBID>.err
    └── rollouts/<exp>/<step>.jsonl
```
