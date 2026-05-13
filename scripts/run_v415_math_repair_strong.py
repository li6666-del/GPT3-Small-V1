from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import chat_row, eval_row, write_jsonl  # noqa: E402
from scripts.run_v415_ability_mustfix import (  # noqa: E402
    ABILITY_ROWS,
    ABILITY_STAGE_RULES,
    CORE_QA_ROWS,
    ENGLISH_OBSERVE,
    IDENTITY_ROWS,
    LOCKED_MAIN_RULES,
    MATH_REGRESSION,
    MODEL,
    SAFETY_ROWS,
    STOP_ROWS,
    UNKNOWN_ROWS,
    q,
    round_rules,
    summarize,
)
from scripts.run_v415_anchor_repair import prune_temp  # noqa: E402
from scripts.run_v415_canhelp_retry import CAN_HELP_ROWS  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


Row = tuple[str, str, str, str]

ONE_PLUS_FOUR_ROWS: list[Row] = [
    ("1 加 4 等于多少？", "1 + 4 = 5。", "math_one_plus_four", "zh"),
    ("请计算 1 加 4。", "1 + 4 = 5。", "math_one_plus_four", "zh"),
    ("一加四等于几？", "1 + 4 = 5。", "math_one_plus_four", "zh"),
    ("1+4 等于多少？", "1 + 4 = 5。", "math_one_plus_four", "zh"),
    ("1 与 4 相加是多少？", "1 + 4 = 5。", "math_one_plus_four", "zh"),
    ("4 加 1 等于多少？", "4 + 1 = 5。", "math_one_plus_four", "zh"),
]

WEEK_ROWS: list[Row] = [
    ("一周有几天？", "一周有 7 天。", "week_anchor", "zh"),
    ("一星期有几天？", "一星期有 7 天。", "week_anchor", "zh"),
    ("一周通常是多少天？", "一周通常是 7 天。", "week_anchor", "zh"),
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def sample_rows(pool: list[Row], count: int, seed: int, source: str) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = [chat_row(*rng.choice(pool), source=source) for _ in range(count)]
    rng.shuffle(rows)
    return rows


def eval_prompts() -> list[dict[str, object]]:
    rows = (
        IDENTITY_ROWS
        + STOP_ROWS
        + UNKNOWN_ROWS
        + SAFETY_ROWS
        + CORE_QA_ROWS
        + ONE_PLUS_FOUR_ROWS
        + WEEK_ROWS
        + MATH_REGRESSION
        + ABILITY_ROWS
        + CAN_HELP_ROWS
        + ENGLISH_OBSERVE
    )
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for prompt, response, category, language in rows:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(eval_row(len(out), prompt, response, category, language, "v415_math_repair_eval"))
    return out


def build_data() -> str:
    source = "synthetic_v415_math_repair_strong"
    out_dir = REPO_ROOT / "data/sft/v415/03_math_repair_strong"
    pool = (
        ONE_PLUS_FOUR_ROWS * 360
        + WEEK_ROWS * 120
        + ABILITY_ROWS * 58
        + CAN_HELP_ROWS * 58
        + UNKNOWN_ROWS * 35
        + (IDENTITY_ROWS + STOP_ROWS + CORE_QA_ROWS[:2] + [MATH_REGRESSION[0]]) * 14
        + SAFETY_ROWS * 3
        + ENGLISH_OBSERVE
    )
    train = sample_rows(pool, 4300, 20261401, source)
    valid = sample_rows(pool, 430, 20261402, source)
    prompts = eval_prompts()
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_prompts.jsonl", prompts)
    write_json(
        out_dir / "manifest.json",
        {
            "train_examples": len(train),
            "valid_examples": len(valid),
            "eval_prompts": len(prompts),
            "train_distribution": summarize(train),
            "valid_distribution": summarize(valid),
        },
    )
    print(f"[v415-math] built {len(train)} train / {len(valid)} valid / {len(prompts)} eval")
    print(f"[v415-math] distribution={summarize(train)}")
    return "data/sft/v415/03_math_repair_strong"


def build_config(data_dir: str, init_checkpoint: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v415_03_math_repair_strong.json"
    lr = 1.25e-6
    config = {
        "run_name": "sft-v415-03-math_repair_strong",
        "out_dir": "runs/sft-v415-03-math_repair_strong",
        "init_checkpoint": init_checkpoint,
        "seed": 20261411,
        "device": "auto",
        "dtype": "bfloat16",
        "compile": False,
        "model": MODEL,
        "data": {
            "train_path": f"{data_dir}/train.jsonl",
            "valid_path": f"{data_dir}/valid.jsonl",
            "tokenizer_json_path": "artifacts/tokenizer/tokenizer.json",
            "vocab_path": "artifacts/tokenizer/vocab.bin",
            "merges_path": "artifacts/tokenizer/merges.bin",
            "batch_size": 8,
            "train_eot": True,
            "truncate": True,
        },
        "optim": {
            "learning_rate": lr,
            "min_lr": lr * 0.1,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_clip": 1.0,
        },
        "train": {
            "max_steps": 52,
            "warmup_steps": 5,
            "gradient_accumulation_steps": 8,
            "log_interval": 4,
            "eval_interval": 4,
            "eval_iters": 12,
            "save_interval": 4,
            "resume": True,
            "generation_eval": {
                "enabled": True,
                "prompts_path": f"{data_dir}/eval_prompts.jsonl",
                "output_path": "generation_eval.jsonl",
                "interval": 4,
                "max_new_tokens": 50,
                "temperature": 0.35,
                "top_k": 50,
                "stop_at_eot": True,
                "seed": 20261411,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261411,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return "configs/sft_125m_v415_03_math_repair_strong.json"


def build_experiment(data_dir: str, config_path: str) -> str:
    experiment_path = REPO_ROOT / "experiments/v415/sft_v415_03_math_repair_strong.yaml"
    experiment = {
        "name": "sft-v415-03-math_repair_strong",
        "local_root": ".",
        "remote": {
            "host": "connect.bjb2.seetacloud.com",
            "port": 52387,
            "user": "root",
            "password_env": "AUTODL_PASSWORD",
            "project_dir": "/root/autodl-tmp/GPT3-small-V1",
            "python": "/root/miniconda3/bin/python",
        },
        "upload": {"items": [{"local": config_path}, {"local": data_dir}]},
        "train": {
            "config": config_path,
            "run_dir": "runs/sft-v415-03-math_repair_strong",
            "pid_file": "logs/sft-v415-03-math_repair_strong.pid",
            "stdout": "logs/sft-v415-03-math_repair_strong.stdout",
            "stderr": "logs/sft-v415-03-math_repair_strong.stderr",
            "fresh": True,
            "clear_run_dir": True,
        },
        "monitor": {
            "interval_sec": 40,
            "max_minutes": 80,
            "min_failure_step": 999,
            "kill_on_failure": True,
        },
        "evaluation": {
            "generation_eval_path": "runs/sft-v415-03-math_repair_strong/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": round_rules(ABILITY_STAGE_RULES, main_names=LOCKED_MAIN_RULES),
        },
        "cleanup": {
            "enabled": True,
            "run_dir": "runs/sft-v415-03-math_repair_strong",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": "reports/sft/v415/sft-v415-03-math_repair_strong.md",
            "cache_dir": "reports/sft/v415/sft-v415-03-math_repair_strong",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    return str(experiment_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", default="runs/sft-v415-01-canhelp_retry/step_000048.pt")
    args = parser.parse_args()
    if not os.environ.get("AUTODL_PASSWORD"):
        raise SystemExit("AUTODL_PASSWORD is required")
    data_dir = build_data()
    config_path = build_config(data_dir, args.initial_checkpoint)
    experiment_path = build_experiment(data_dir, config_path)
    result = run_once(Path(experiment_path).resolve())
    final = None
    cleanup_note = "not_passed"
    if result.status == "passed" and result.selected_step is not None:
        final = f"runs/sft-v415-03-math_repair_strong/step_{int(result.selected_step):06d}.pt"
        cleanup_note = prune_temp(
            {
                "host": "connect.bjb2.seetacloud.com",
                "port": 52387,
                "user": "root",
                "password_env": "AUTODL_PASSWORD",
                "project_dir": "/root/autodl-tmp/GPT3-small-V1",
            },
            final,
        )
    print(json.dumps(result.__dict__ | {"report_path": str(result.report_path), "final_checkpoint": final, "cleanup": cleanup_note}, ensure_ascii=False, indent=2))
    if result.status == "failed":
        sys.exit(2)


if __name__ == "__main__":
    main()
