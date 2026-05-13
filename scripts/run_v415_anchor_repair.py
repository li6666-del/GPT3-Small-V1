from __future__ import annotations

import argparse
import json
import os
import random
import shlex
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
    SHORT_QA_CORRECTIONS,
    STOP_ROWS,
    UNKNOWN_ROWS,
    q,
    round_rules,
    summarize,
)
from scripts.run_v415_canhelp_retry import CAN_HELP_ROWS  # noqa: E402
from scripts.sft_harness import RemoteSession, run_once  # noqa: E402


Row = tuple[str, str, str, str]


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
        + SHORT_QA_CORRECTIONS
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
        out.append(eval_row(len(out), prompt, response, category, language, "v415_anchor_eval"))
    return out


def build_data() -> str:
    source = "synthetic_v415_anchor_repair"
    out_dir = REPO_ROOT / "data/sft/v415/02_anchor_repair"
    pool = (
        SHORT_QA_CORRECTIONS * 170
        + UNKNOWN_ROWS * 42
        + ABILITY_ROWS * 36
        + CAN_HELP_ROWS * 36
        + (IDENTITY_ROWS + STOP_ROWS + CORE_QA_ROWS[:2] + [MATH_REGRESSION[0]]) * 18
        + SAFETY_ROWS * 3
        + ENGLISH_OBSERVE
    )
    train = sample_rows(pool, 4000, 20261301, source)
    valid = sample_rows(pool, 420, 20261302, source)
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
    print(f"[v415-anchor] built {len(train)} train / {len(valid)} valid / {len(prompts)} eval")
    print(f"[v415-anchor] distribution={summarize(train)}")
    return "data/sft/v415/02_anchor_repair"


def build_config(data_dir: str, init_checkpoint: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v415_02_anchor_repair.json"
    lr = 4.5e-7
    config = {
        "run_name": "sft-v415-02-anchor_repair",
        "out_dir": "runs/sft-v415-02-anchor_repair",
        "init_checkpoint": init_checkpoint,
        "seed": 20261311,
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
            "max_steps": 56,
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
                "seed": 20261311,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261311,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return "configs/sft_125m_v415_02_anchor_repair.json"


def build_experiment(data_dir: str, config_path: str) -> str:
    experiment_path = REPO_ROOT / "experiments/v415/sft_v415_02_anchor_repair.yaml"
    experiment = {
        "name": "sft-v415-02-anchor_repair",
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
            "run_dir": "runs/sft-v415-02-anchor_repair",
            "pid_file": "logs/sft-v415-02-anchor_repair.pid",
            "stdout": "logs/sft-v415-02-anchor_repair.stdout",
            "stderr": "logs/sft-v415-02-anchor_repair.stderr",
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
            "generation_eval_path": "runs/sft-v415-02-anchor_repair/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": round_rules(ABILITY_STAGE_RULES, main_names=LOCKED_MAIN_RULES),
        },
        "cleanup": {
            "enabled": True,
            "run_dir": "runs/sft-v415-02-anchor_repair",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": "reports/sft/v415/sft-v415-02-anchor_repair.md",
            "cache_dir": "reports/sft/v415/sft-v415-02-anchor_repair",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    return str(experiment_path)


def prune_temp(remote_cfg: dict[str, Any], keep: str) -> str:
    project_dir = str(remote_cfg["project_dir"])
    command = f"""
set -e
cd {q(project_dir)}
keep={q(keep)}
for dir in runs/sft-v415-00-ability_acquire_balanced runs/sft-v415-01-canhelp_retry; do
  [ -d "$dir" ] || continue
  for path in "$dir"/*.pt; do
    [ -e "$path" ] || continue
    rm -f "$path"
    echo deleted:$path
  done
done
for path in runs/sft-v415-02-anchor_repair/*.pt; do
  [ -e "$path" ] || continue
  if [ "$path" != "$keep" ]; then
    rm -f "$path"
    echo deleted:$path
  else
    echo kept:$path
  fi
done
"""
    remote = RemoteSession(remote_cfg)
    try:
        rc, out, err = remote.run(command, timeout=180)
        if rc != 0:
            return f"prune_failed:{err.strip()}"
        return out.strip() or "nothing_to_prune"
    finally:
        remote.close()


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
    if result.status == "passed" and result.selected_step is not None:
        final = f"runs/sft-v415-02-anchor_repair/step_{int(result.selected_step):06d}.pt"
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
    else:
        cleanup_note = "final_not_passed_temp_kept_for_debug"
    print(json.dumps(result.__dict__ | {"report_path": str(result.report_path), "final_checkpoint": final, "cleanup": cleanup_note}, ensure_ascii=False, indent=2))
    if result.status == "failed":
        sys.exit(2)


if __name__ == "__main__":
    main()
