"""
IDK abstention reward function.

Scoring:
  +1.0  model answer is correct (verified by math-verify)
  +0.5  model emits \\boxed{idk}
   0.0  model answer is wrong (or no boxed answer found)

Returns a dict so VERL auto-logs per-step means to wandb:
  reward/score      — mean reward (includes 0.5s)
  reward/is_idk     — idk_rate  (fraction of batch that said idk)
  reward/is_correct — success_rate (fraction that got it right)

Uses math-verify for symbolic equivalence checking (no API calls).
Based on maxrl's MathVerifyScorer with thread-safety fallback.
Install: pip install math-verify
"""

import re

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


class MathVerifyScorer:
    """Thread-safe math-verify scorer for Ray workers.

    math-verify uses signal.alarm() internally which fails in non-main
    threads (Ray workers). When that happens, falls back to exact string
    match on extracted \\boxed{} content.
    """

    def __init__(self):
        self._verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        self._use_fallback = False

    def check_correct(self, model_output: str, ground_truth: str) -> float:
        ground_truth = str(ground_truth)
        ground_truth_boxed = f"\\boxed{{{ground_truth}}}"

        if not self._use_fallback:
            try:
                score, _ = self._verify_func([ground_truth_boxed], [model_output])
                return float(score)
            except TimeoutException:
                return 0.0
            except ValueError as e:
                if "threaded" in str(e) or "signal" in str(e):
                    self._use_fallback = True
                    print("[reward_fn] math-verify threading error, using exact match fallback")
                else:
                    return 0.0
            except Exception:
                return 0.0

        # Fallback: exact string match on extracted boxed content
        pred_boxed = _extract_boxed(model_output)
        if pred_boxed is not None and pred_boxed.strip() == ground_truth.strip():
            return 1.0
        return 0.0


# Lazy singleton — one per Ray worker process
_scorer = None


def _get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = MathVerifyScorer()
    return _scorer


def _extract_boxed(text: str) -> str | None:
    """Return content of the last \\boxed{...} in text, or None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = re.search(r"\\boxed\s+(\S+)", text)
        return m.group(1).strip() if m else None
    depth = 0
    for i in range(idx + len("\\boxed{") - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx + len("\\boxed{") : i].strip()
    return None


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """Synchronous reward function compatible with NaiveRewardManager."""
    boxed = _extract_boxed(solution_str)

    if boxed is None:
        return {"score": 0.0, "is_idk": 0.0, "is_correct": 0.0}

    if boxed.lower() == "idk":
        return {"score": 0.5, "is_idk": 1.0, "is_correct": 0.0}

    correct = _get_scorer().check_correct(solution_str, ground_truth)
    return {"score": correct, "is_idk": 0.0, "is_correct": correct}
