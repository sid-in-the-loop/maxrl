"""
Unified preprocessing script for IDK abstention datasets.

Usage:
  python preprocess.py --dataset math12k [--idk] [--local_dir PATH] [--src PATH]
  python preprocess.py --dataset math500  [--idk] [--local_dir PATH]
  python preprocess.py --dataset polaris  [--idk] [--local_dir PATH]

--idk   adds the abstention suffix: "If you feel even slightly uncertain, put \\boxed{idk}."
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

BASE_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
IDK_SUFFIX = " If you feel even slightly uncertain about the answer, put \\boxed{{idk}}."

HF_CONFIGS = {
    "math500": {
        "hf_name": "HuggingFaceH4/MATH-500",
        "hf_split": "test",
        "data_source_name": "DigitalLearningGmbH/MATH-lighteval",
    },
    "polaris": {
        "hf_name": "POLARIS-Project/Polaris-Dataset-53K",
        "hf_split": "train",
        "data_source_name": "polaris",
    },
}


def build_instruction(idk: bool) -> str:
    return BASE_INSTRUCTION + (IDK_SUFFIX if idk else "")


def process_math12k(args):
    src = args.src or "/home/ssmurali/maxrl/data/math12k_train.json"
    local_dir = args.local_dir or f"data/math12k{'_idk' if args.idk else ''}"
    instruction = build_instruction(args.idk)

    with open(src) as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} samples from {src}")

    records = []
    for idx, item in enumerate(raw):
        question = item["question"] + instruction
        answer = str(item["ground_truth"])
        if idx == 0:
            print("Question:", question[:200])
            print("Answer:", answer)
        records.append({
            "data_source": "math12k",
            "id": idx,
            "prompt": [{"role": "user", "content": question}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": "train", "index": idx},
        })

    os.makedirs(local_dir, exist_ok=True)
    table = pa.table({k: [r[k] for r in records] for k in records[0]})
    out = os.path.join(local_dir, "train.parquet")
    pq.write_table(table, out)
    print(f"Saved {len(records)} samples to {out}")


def process_hf(args):
    import datasets as ds

    cfg = HF_CONFIGS[args.dataset]
    local_dir = os.path.expanduser(
        args.local_dir or f"~/data/{args.dataset}{'_idk' if args.idk else ''}"
    )
    instruction = build_instruction(args.idk)
    data_source_name = cfg["data_source_name"]

    print(f"Loading {cfg['hf_name']} from HuggingFace...", flush=True)
    dataset = ds.load_dataset(cfg["hf_name"], "default", trust_remote_code=True)
    split = cfg["hf_split"]
    train_dataset = dataset[split]
    test_dataset = dataset[split]

    def make_map_fn(split_name):
        def process_fn(example, idx):
            question = example.pop("problem") + instruction
            answer = str(example.pop("answer"))
            if idx == 0:
                print("Question:", question[:200])
                print("Answer:", answer)
            return {
                "data_source": data_source_name,
                "id": idx,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split_name, "index": idx},
            }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    print(f"Saved to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["math12k", "math500", "polaris"])
    parser.add_argument("--idk", action="store_true", help="Add IDK abstention instruction")
    parser.add_argument("--local_dir", default=None, help="Output directory (default: ~/data/<dataset>[_idk])")
    parser.add_argument("--src", default=None, help="math12k only: path to math12k_train.json")
    args = parser.parse_args()

    if args.dataset == "math12k":
        process_math12k(args)
    else:
        process_hf(args)
