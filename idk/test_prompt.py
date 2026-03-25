"""
Quick experiment: load prompts from parquet and test on Qwen3-1.7B-Base.
Measures accuracy via math-verify and prints sample outputs.
Uses vLLM for fast batched inference.

Usage:
  python idk/test_prompt.py [--n_samples 50] [--data data/math12k_train.parquet]
  python idk/test_prompt.py --chat_template   # apply chat template (simulates verl pipeline)
"""

import re
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from math_verify.errors import TimeoutException

# ── Verification ─────────────────────────────────────────────────────────────

_verify = math_metric(
    gold_extraction_target=(LatexExtractionConfig(),),
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
)


def check_correct(model_output: str, ground_truth: str) -> float:
    gt_boxed = "\\boxed{" + ground_truth + "}"
    try:
        score, _ = _verify([gt_boxed], [model_output])
        return float(score)
    except TimeoutException:
        return 0.0
    except BaseException:
        return 0.0


# ── Parquet helpers ──────────────────────────────────────────────────────────

def _extract_content(prompt_data):
    """Extract plain text content from various parquet prompt formats."""
    # Try direct access as list/dict
    try:
        if hasattr(prompt_data, '__iter__') and not isinstance(prompt_data, str):
            first = prompt_data[0]
            # Could be dict, pyarrow struct, or pandas object
            if hasattr(first, 'get'):
                return first.get('content', str(first))
            elif hasattr(first, '__getitem__'):
                return first['content']
    except (KeyError, IndexError, TypeError):
        pass
    # Try parsing as string repr
    s = str(prompt_data)
    m = re.search(r"'content':\s*'(.*?)'(?:,\s*'role'|\})", s, re.DOTALL)
    if not m:
        m = re.search(r'"content":\s*"(.*?)"(?:,\s*"role"|\})', s, re.DOTALL)
    if m:
        return m.group(1)
    return s


def _extract_gt(reward_info):
    """Extract ground truth from reward_model field."""
    try:
        if hasattr(reward_info, 'get'):
            return reward_info.get('ground_truth', str(reward_info))
        elif hasattr(reward_info, '__getitem__'):
            return reward_info['ground_truth']
    except (KeyError, TypeError):
        pass
    s = str(reward_info)
    import re
    m = re.search(r"'ground_truth':\s*'(.*?)'", s)
    if not m:
        m = re.search(r'"ground_truth":\s*"(.*?)"', s)
    if m:
        return m.group(1)
    return s


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--data", default="/home/ssmurali/maxrl/data/polaris_train.parquet")
    parser.add_argument("--chat_template", action="store_true",
                        help="Apply chat template to prompts (simulates what verl does)")
    args = parser.parse_args()

    # Load parquet
    df = pd.read_parquet(args.data)
    df = df.head(args.n_samples)
    print(f"Loaded {len(df)} samples from {args.data}")

    # Extract prompts and ground truths from parquet
    prompts = []
    chat_messages = []  # raw chat format for --chat_template mode
    gts = []
    for _, row in df.iterrows():
        prompt_data = row["prompt"]
        # prompt_data is a list of message dicts (possibly with few-shot examples)
        if isinstance(prompt_data, list) and len(prompt_data) > 0 and isinstance(prompt_data[0], dict):
            chat_messages.append(prompt_data)
            content = prompt_data[-1].get("content", str(prompt_data[-1]))
        else:
            content = _extract_content(prompt_data)
            chat_messages.append([{"role": "user", "content": content}])
        prompts.append(content)

        reward_info = row["reward_model"]
        gt = _extract_gt(reward_info)
        gts.append(gt)

    # If --chat_template, apply the tokenizer's chat template (simulates verl)
    if args.chat_template:
        MODEL = "Qwen/Qwen3-1.7B-Base"
        print(f"Applying chat template from {MODEL} tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        prompts = [
            tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in chat_messages
        ]
        print(f"Mode: CHAT TEMPLATE (what verl does)")
    else:
        print(f"Mode: PLAIN TEXT (no chat template)")

    # Show first prompt
    print(f"\n--- Sample prompt ---\n{prompts[0][:500]}\n--- end ---\n")

    print("Loading vLLM engine...")
    llm = LLM(
        model="Qwen/Qwen3-1.7B-Base",
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=5120,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(f"Generating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params)

    correct = 0
    total = len(outputs)
    for i, (output, gt) in enumerate(zip(outputs, gts)):
        response = output.outputs[0].text
        score = check_correct(response, gt)
        correct += score

        if i < 10:
            resp_preview = response.replace("\n", " | ")
            print(f"\n  [{i}] GT={gt}")
            print(f"      Score={score}")
            print(f"      Response: {resp_preview}")

    acc = correct / total if total > 0 else 0
    print(f"\n  >> {correct:.0f}/{total} = {acc:.1%}")
    print("\nDone.")


if __name__ == "__main__":
    main()
