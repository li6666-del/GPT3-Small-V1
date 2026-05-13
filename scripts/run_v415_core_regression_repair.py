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
    ALL_RULES,
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
from scripts.run_v415_math_repair_strong import ONE_PLUS_FOUR_ROWS, WEEK_ROWS  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


Row = tuple[str, str, str, str]

CORE_REPAIR_ROWS: list[Row] = [
    ("水在标准大气压下通常多少摄氏度沸腾？", "水在标准大气压下通常约 100 摄氏度沸腾。", "core_regression_repair", "zh"),
    ("标准大气压下水通常多少度沸腾？", "标准大气压下水通常约 100 摄氏度沸腾。", "core_regression_repair", "zh"),
    ("9 减 4 等于多少？", "9 - 4 = 5。", "core_regression_repair", "zh"),
    ("请计算 9 减 4。", "9 - 4 = 5。", "core_regression_repair", "zh"),
    ("10 减 7 等于多少？", "10 - 7 = 3。", "core_regression_repair", "zh"),
    ("5 乘以 8 等于多少？", "5 × 8 = 40。", "core_regression_repair", "zh"),
    ("4 乘以 6 等于多少？", "4 × 6 = 24。", "core_regression_repair", "zh"),
]

EXTRA_RULES: dict[str, dict[str, Any]] = {
    "simple_qa_boiling_exact": {
        "name": "simple_qa_boiling_exact",
        "layer": "main",
        "mode": "greedy",
        "prompt": "水在标准大气压下通常多少摄氏度沸腾？",
        "must_include_any": ["100", "一百"],
        "must_not_include_any": ["15分钟", "15 分钟"],
        "max_chars": 60,
    },
    "math_sub_9_4_exact": {
        "name": "math_sub_9_4_exact",
        "layer": "main",
        "mode": "greedy",
        "prompt": "9 减 4 等于多少？",
        "must_include_all": ["9 - 4 = 5"],
        "must_not_include_any": ["9 + 4"],
        "max_chars": 40,
    },
    "math_sub_10_7_exact": {
        "name": "math_sub_10_7_exact",
        "layer": "main",
        "mode": "greedy",
        "prompt": "10 减 7 等于多少？",
        "must_include_all": ["10 - 7 = 3"],
        "max_chars": 40,
    },
}

ALL_RULES.update(EXTRA_RULES)
EXTENDED_MAIN = LOCKED_MAIN_RULES + ["simple_qa_boiling_exact", "math_sub_9_4_exact", "math_sub_10_7_exact"]


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
        + CORE_REPAIR_ROWS
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
        out.append(eval_row(len(out), prompt, response, category, language, "v415_core_repair_eval"))
    return out


def build_data() -> str:
    source = "synthetic_v415_core_regression_repair"
    out_dir = REPO_ROOT / "data/sft/v415/04_core_regression_repair"
    pool = (
        CORE_REPAIR_ROWS * 120
        + ONE_PLUS_FOUR_ROWS * 145
        + WEEK_ROWS * 80
        + ABILITY_ROWS * 58
        + CAN_HELP_ROWS * 58
        + UNKNOWN_ROWS * 34
        + (IDENTITY_ROWS + STOP_ROWS + CORE_QA_ROWS[:2] + [MATH_REGRESSION[0]]) * 14
        + SAFETY_ROWS * 3
        + ENGLISH_OBSERVE
    )
    train = sample_rows(pool, 4300, 20261501, source)
    valid = sample_rows(pool, 430, 20261502, source)
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
    print(f"[v415-core] built {len(train)} train / {len(valid)} valid / {len(prompts)} eval")
    print(f"[v415-core] distribution={summarize(train)}")
    return "data/sft/v415/04_core_regression_repair"


def build_config(data_dir: str, init_checkpoint: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v415_04_core_regression_repair.json"
    lr = 4.0e-7
    config = {
        "run_name": "sft-v415-04-core_regression_repair",
        "out_dir": "runs/sft-v415-04-core_regression_repair",
        "init_checkpoint": init_checkpoint,
        "seed": 20261511,
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
            "max_steps": 44,
            "warmup_steps": 4,
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
                "seed": 20261511,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261511,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return "configs/sft_125m_v415_04_core_regression_repair.json"


def build_experiment(data_dir: str, config_path: str) -> str:
    experiment_path = REPO_ROOT / "experiments/v415/sft_v415_04_core_regression_repair.yaml"
    experiment = {
        "name": "sft-v415-04-core_regression_repair",
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
            "run_dir": "runs/sft-v415-04-core_regression_repair",
            "pid_file": "logs/sft-v415-04-core_regression_repair.pid",
            "stdout": "logs/sft-v415-04-core_regression_repair.stdout",
            "stderr": "logs/sft-v415-04-core_regression_repair.stderr",
            "fresh": True,
            "clear_run_dir": True,
        },
        "monitor": {
            "interval_sec": 40,
            "max_minutes": 75,
            "min_failure_step": 999,
            "kill_on_failure": True,
        },
        "evaluation": {
            "generation_eval_path": "runs/sft-v415-04-core_regression_repair/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": round_rules(ABILITY_STAGE_RULES, main_names=EXTENDED_MAIN),
        },
        "cleanup": {
            "enabled": True,
            "run_dir": "runs/sft-v415-04-core_regression_repair",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": "reports/sft/v415/sft-v415-04-core_regression_repair.md",
            "cache_dir": "reports/sft/v415/sft-v415-04-core_regression_repair",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    return str(experiment_path)


def cleanup_old(final: str) -> str:
    return prune_temp(
        {
            "host": "connect.bjb2.seetacloud.com",
            "port": 52387,
            "user": "root",
            "password_env": "AUTODL_PASSWORD",
            "project_dir": "/root/autodl-tmp/GPT3-small-V1",
        },
        final,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", default="runs/sft-v415-03-math_repair_strong/step_000051.pt")
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
        final = f"runs/sft-v415-04-core_regression_repair/step_{int(result.selected_step):06d}.pt"
        cleanup_note = cleanup_old(final)
    print(json.dumps(result.__dict__ | {"report_path": str(result.report_path), "final_checkpoint": final, "cleanup": cleanup_note}, ensure_ascii=False, indent=2))
    if result.status == "failed":
        sys.exit(2)


if __name__ == "__main__":
    main()
