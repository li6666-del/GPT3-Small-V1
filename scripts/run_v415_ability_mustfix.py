from __future__ import annotations

import argparse
import json
import os
import posixpath
import random
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import chat_row, eval_row, write_jsonl  # noqa: E402
from scripts.eval_sft_outputs import complete_steps, count_jsonl, evaluate_rows, load_jsonl  # noqa: E402
from scripts.run_v414_preheldout_loop import (  # noqa: E402
    CORE_QA_ROWS,
    ENGLISH_OBSERVE,
    IDENTITY_ROWS,
    MAIN_RULES,
    MATH_REGRESSION,
    MODEL,
    RULES,
    SAFETY_ROWS,
    SHORT_QA_CORRECTIONS,
    STOP_ROWS,
    UNKNOWN_ROWS,
    q,
    summarize,
    write_json,
    write_yaml,
)
from scripts.sft_harness import RemoteSession, run_once  # noqa: E402


Row = tuple[str, str, str, str]


ABILITY_ROWS: list[Row] = [
    ("你能做什么？", "我可以回答简单问题、做简短解释；不确定时会说明不能确定。", "ability_plain", "zh"),
    ("你能做什么？", "我可以回答简单问题、做简短解释，也会说明不确定的地方。", "ability_plain", "zh"),
    ("你的能力是什么？", "我可以回答简单问题、做简短解释；复杂事实需要验证。", "ability_plain", "zh"),
    ("你可以帮我做什么？", "我可以回答简单问题、做简短解释，也可以协助整理信息。", "ability_plain", "zh"),
    ("请简短说明你的能力。", "我可以回答简单问题、做简短解释；不确定时会直说。", "ability_plain", "zh"),
    ("你适合做什么？", "我适合回答简单问题、做简短解释和整理信息。", "ability_plain", "zh"),
    ("你不知道的时候怎么办？", "我应该说明不能确定，而不是编造答案。", "ability_plain", "zh"),
    ("你不能做什么？", "我不能保证复杂事实一定正确，也不能帮助危险请求。", "ability_plain", "zh"),
]


ABILITY_STAGE_RULES = [
    "ability_simple_assistant",
    "ability_capability",
    "ability_can_help",
    "ability_brief",
    "ability_limit",
]


EXTRA_RULES: dict[str, dict[str, Any]] = {
    "ability_capability": {
        "name": "ability_capability",
        "layer": "stage",
        "mode": "greedy",
        "prompt": "你的能力是什么？",
        "must_include_all": ["简单问题", "简短解释"],
        "must_not_include_any": ["不能帮助帮助"],
        "max_chars": 90,
    },
    "ability_can_help": {
        "name": "ability_can_help",
        "layer": "stage",
        "mode": "greedy",
        "prompt": "你可以帮我做什么？",
        "must_include_any": ["简单问题", "简短解释", "整理信息"],
        "must_not_include_any": ["可以。"],
        "max_chars": 90,
    },
    "ability_brief": {
        "name": "ability_brief",
        "layer": "stage",
        "mode": "greedy",
        "prompt": "请简短说明你的能力。",
        "must_include_all": ["简单问题", "简短解释"],
        "must_not_include_any": ["不能帮助帮助"],
        "max_chars": 90,
    },
}


ALL_RULES = dict(RULES) | EXTRA_RULES
LOCKED_MAIN_RULES = MAIN_RULES + ["simple_qa_week_days", "math_add_1_4_exact"]


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
        + ENGLISH_OBSERVE
    )
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for prompt, response, category, language in rows:
        if prompt in seen:
            continue
        seen.add(prompt)
        out.append(eval_row(len(out), prompt, response, category, language, "v415_eval"))
    return out


def build_pool(plan: dict[str, Any]) -> list[Row]:
    ability_weight = int(plan.get("ability_weight", 80))
    short_weight = int(plan.get("short_weight", 42))
    unknown_weight = int(plan.get("unknown_weight", 18))
    regression_weight = int(plan.get("regression_weight", 8))
    safety_weight = int(plan.get("safety_weight", 2))
    return (
        ABILITY_ROWS * ability_weight
        + SHORT_QA_CORRECTIONS * short_weight
        + UNKNOWN_ROWS * unknown_weight
        + (IDENTITY_ROWS + STOP_ROWS + CORE_QA_ROWS[:2] + [MATH_REGRESSION[0]]) * regression_weight
        + SAFETY_ROWS * safety_weight
        + ENGLISH_OBSERVE
    )


def build_data(round_index: int, plan: dict[str, Any]) -> str:
    focus = str(plan["focus"])
    source = f"synthetic_v415_ability_mustfix_{round_index:02d}_{focus}"
    out_dir = REPO_ROOT / f"data/sft/v415/{round_index:02d}_{focus}"
    pool = build_pool(plan)
    train = sample_rows(pool, int(plan["train_examples"]), 20260900 + round_index, source)
    valid = sample_rows(pool, int(plan["valid_examples"]), 20261000 + round_index, source)
    prompts = eval_prompts()
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_prompts.jsonl", prompts)
    write_json(
        out_dir / "manifest.json",
        {
            "round": round_index,
            "plan": plan,
            "train_examples": len(train),
            "valid_examples": len(valid),
            "eval_prompts": len(prompts),
            "train_distribution": summarize(train),
            "valid_distribution": summarize(valid),
        },
    )
    print(f"[v415] built {len(train)} train / {len(valid)} valid / {len(prompts)} eval for {focus}")
    print(f"[v415] distribution={summarize(train)}")
    return f"data/sft/v415/{round_index:02d}_{focus}"


def round_rules(stage_names: list[str], main_names: list[str] | None = None) -> list[dict[str, Any]]:
    main = main_names or LOCKED_MAIN_RULES
    names = main + stage_names + ["simple_qa_english_sky"]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        rule = dict(ALL_RULES[name])
        if name in main:
            rule["layer"] = "main"
        elif name == "simple_qa_english_sky":
            rule["layer"] = "observe"
        else:
            rule["layer"] = "stage"
        out.append(rule)
    return out


def build_config(run_name: str, data_dir: str, init_checkpoint: str, plan: dict[str, Any], seed: int) -> dict[str, Any]:
    lr = float(plan["lr"])
    return {
        "run_name": run_name,
        "out_dir": f"runs/{run_name}",
        "init_checkpoint": init_checkpoint,
        "seed": seed,
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
            "max_steps": int(plan["max_steps"]),
            "warmup_steps": int(plan.get("warmup_steps", 6)),
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
                "seed": seed,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": seed,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }


def build_experiment(
    run_name: str,
    data_dir: str,
    config_path: str,
    plan: dict[str, Any],
    cleanup_enabled: bool,
    main_names: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": run_name,
        "local_root": ".",
        "remote": {
            "host": "connect.bjb2.seetacloud.com",
            "port": 52387,
            "user": "root",
            "password_env": "AUTODL_PASSWORD",
            "project_dir": "/root/autodl-tmp/GPT3-small-V1",
            "python": "/root/miniconda3/bin/python",
        },
        "upload": {
            "items": [
                {"local": config_path},
                {"local": data_dir},
            ]
        },
        "train": {
            "config": config_path,
            "run_dir": f"runs/{run_name}",
            "pid_file": f"logs/{run_name}.pid",
            "stdout": f"logs/{run_name}.stdout",
            "stderr": f"logs/{run_name}.stderr",
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
            "generation_eval_path": f"runs/{run_name}/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": round_rules(list(plan["rules"]), main_names=main_names),
        },
        "cleanup": {
            "enabled": cleanup_enabled,
            "run_dir": f"runs/{run_name}",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": f"reports/sft/v415/{run_name}.md",
            "cache_dir": f"reports/sft/v415/{run_name}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }


def write_artifacts(round_index: int, plan: dict[str, Any], init_checkpoint: str, cleanup_enabled: bool) -> tuple[str, dict[str, Any]]:
    focus = str(plan["focus"])
    run_name = f"sft-v415-{round_index:02d}-{focus}"
    data_dir = build_data(round_index, plan)
    config_path = f"configs/sft_125m_v415_{round_index:02d}_{focus}.json"
    experiment_path = REPO_ROOT / f"experiments/v415/sft_v415_{round_index:02d}_{focus}.yaml"
    seed = 20261100 + round_index
    config = build_config(run_name, data_dir, init_checkpoint, plan, seed)
    write_json(REPO_ROOT / config_path, config)
    experiment = build_experiment(run_name, data_dir, config_path, plan, cleanup_enabled)
    write_yaml(experiment_path, experiment)
    memo = REPO_ROOT / f"reports/sft/v415/strategy_{round_index:02d}_{focus}.md"
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(
        "\n".join(
            [
                f"# V4.15 Round {round_index:02d} Strategy",
                "",
                f"- focus: `{focus}`",
                f"- init_checkpoint: `{init_checkpoint}`",
                f"- learning_rate: `{plan['lr']}`",
                f"- max_steps: `{plan['max_steps']}`",
                f"- cleanup_enabled: `{cleanup_enabled}`",
                f"- stage_rules: `{', '.join(plan['rules'])}`",
                "",
                "目标：能力说明必须通过，同时守住 V4.14 已修复的 `1+4` 和一周七天。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(experiment_path), experiment


def ability_candidate_from_report(experiment: dict[str, Any]) -> tuple[int | None, dict[str, Any] | None]:
    cache = REPO_ROOT / str(experiment["report"]["cache_dir"])
    generation = cache / "generation_eval.jsonl"
    prompts = REPO_ROOT / str(experiment["evaluation"]["prompts_path"])
    rows = load_jsonl(generation)
    complete = complete_steps(rows, count_jsonl(prompts), ["greedy"])
    candidates: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for step in complete:
        result = evaluate_rows(
            rows,
            list(experiment["evaluation"]["rules"]),
            expected_prompts=count_jsonl(prompts),
            required_modes=["greedy"],
            step=step,
        )
        stage_failed = [rule["name"] for rule in result.get("stage_failed", [])]
        if stage_failed:
            continue
        main_failed = [rule["name"] for rule in result.get("main_failed", [])]
        # Prefer a candidate that has acquired ability and only damaged narrow repairable anchors.
        bad_main = [name for name in main_failed if name not in {"math_add_1_4_exact", "unknown_checkpoint_certain"}]
        if bad_main:
            continue
        score = (len(main_failed), len(result.get("observe_failed", [])), -int(step), 0)
        candidates.append((score, result))
    if not candidates:
        return None, None
    _score, best = min(candidates, key=lambda item: item[0])
    return int(best["selected_step"]), best


def prune_remote(remote_cfg: dict[str, Any], keep: set[str], run_prefixes: list[str]) -> str:
    project_dir = str(remote_cfg["project_dir"])
    keep_expr = " ".join(q(path) for path in sorted(keep))
    prefix_expr = " ".join(q(prefix) for prefix in run_prefixes)
    command = f"""
set -e
cd {q(project_dir)}
keep="{keep_expr}"
for prefix in {prefix_expr}; do
  for path in "$prefix"/*.pt; do
    [ -e "$path" ] || continue
    save=no
    for item in $keep; do
      if [ "$path" = "$item" ]; then save=yes; fi
    done
    if [ "$save" = no ]; then
      rm -f "$path"
      echo deleted:$path
    else
      echo kept:$path
    fi
  done
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


def write_summary(rows: list[dict[str, Any]], kept: set[str]) -> None:
    lines = [
        "# V4.15 Ability Must-Fix Summary",
        "",
        "目标：能力说明必须修好，同时守住身份、stop、拒答、unknown、一周七天和 `1+4`。",
        "",
        "## Rounds",
        "",
        "| round | focus | status | accepted | selected | init_after | summary |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['focus']} | {row['status']} | {row['accepted']} | "
            f"{row.get('selected_step')} | `{row.get('init_after')}` | {row.get('summary')} |"
        )
    lines.extend(["", "## Kept Checkpoints", ""])
    lines.extend([f"- `{path}`" for path in sorted(kept)] or ["- none"])
    path = REPO_ROOT / "reports/sft/v415/adaptive_loop_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    write_json(REPO_ROOT / "reports/sft/v415/adaptive_loop_summary.json", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-checkpoint", default="runs/sft-v414-00-short_qa_corrections/step_000036.pt")
    args = parser.parse_args()
    if not os.environ.get("AUTODL_PASSWORD"):
        raise SystemExit("AUTODL_PASSWORD is required")

    acquire_plan = {
        "focus": "ability_acquire_balanced",
        "lr": 7.0e-7,
        "max_steps": 64,
        "warmup_steps": 6,
        "train_examples": 3600,
        "valid_examples": 360,
        "ability_weight": 78,
        "short_weight": 52,
        "unknown_weight": 24,
        "regression_weight": 10,
        "safety_weight": 2,
        "rules": ABILITY_STAGE_RULES,
    }
    lock_plan = {
        "focus": "ability_lock_repair",
        "lr": 2.6e-7,
        "max_steps": 44,
        "warmup_steps": 4,
        "train_examples": 3400,
        "valid_examples": 360,
        "ability_weight": 42,
        "short_weight": 90,
        "unknown_weight": 38,
        "regression_weight": 14,
        "safety_weight": 2,
        "rules": ABILITY_STAGE_RULES,
    }
    rows: list[dict[str, Any]] = []
    kept = {args.initial_checkpoint}
    current_init = args.initial_checkpoint
    run_prefixes: list[str] = []

    exp_path, experiment = write_artifacts(0, acquire_plan, current_init, cleanup_enabled=False)
    print(f"[v415] round=0 ability acquire init={current_init}", flush=True)
    result = run_once(Path(exp_path).resolve())
    selected = result.selected_step
    accepted = result.status == "passed" and selected is not None and int(selected) > 0
    checkpoint = None
    if accepted:
        checkpoint = f"runs/{experiment['name']}/step_{int(selected):06d}.pt"
        current_init = checkpoint
        kept.add(checkpoint)
    else:
        temp_step, temp_result = ability_candidate_from_report(experiment)
        if temp_step is not None:
            checkpoint = f"runs/{experiment['name']}/step_{temp_step:06d}.pt"
            current_init = checkpoint
            selected = temp_step
            result.summary = "Temporary ability candidate selected for anchor repair: " + str(temp_result["summary"])
        else:
            result.summary += " No temporary ability candidate found."
    run_prefixes.append(f"runs/{experiment['name']}")
    rows.append(
        {
            "round": 0,
            "focus": acquire_plan["focus"],
            "status": result.status,
            "accepted": accepted,
            "selected_step": selected,
            "summary": result.summary,
            "checkpoint": checkpoint if accepted else None,
            "temp_checkpoint": checkpoint if not accepted else None,
            "init_after": current_init,
            "report": str(result.report_path),
        }
    )
    write_summary(rows, kept)

    if not checkpoint:
        prune_note = prune_remote(experiment["remote"], kept, run_prefixes)
        rows[-1]["prune"] = prune_note
        write_summary(rows, kept)
        print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
        return

    if not accepted:
        exp_path, lock_experiment = write_artifacts(1, lock_plan, current_init, cleanup_enabled=True)
        print(f"[v415] round=1 ability lock init={current_init}", flush=True)
        lock_result = run_once(Path(exp_path).resolve())
        lock_selected = lock_result.selected_step
        lock_accepted = lock_result.status == "passed" and lock_selected is not None and int(lock_selected) > 0
        final_checkpoint = None
        if lock_accepted:
            final_checkpoint = f"runs/{lock_experiment['name']}/step_{int(lock_selected):06d}.pt"
            kept = {args.initial_checkpoint, final_checkpoint}
            current_init = final_checkpoint
        run_prefixes.append(f"runs/{lock_experiment['name']}")
        rows.append(
            {
                "round": 1,
                "focus": lock_plan["focus"],
                "status": lock_result.status,
                "accepted": lock_accepted,
                "selected_step": lock_selected,
                "summary": lock_result.summary,
                "checkpoint": final_checkpoint,
                "init_after": current_init,
                "report": str(lock_result.report_path),
            }
        )

    prune_note = prune_remote(experiment["remote"], kept, run_prefixes)
    rows[-1]["prune"] = prune_note
    write_summary(rows, kept)
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
