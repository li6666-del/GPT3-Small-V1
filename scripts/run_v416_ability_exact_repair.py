from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_sft_v46_dataset import write_jsonl  # noqa: E402
from scripts.run_v416_boundary_repair import CORE_ANCHOR_ROWS, sample_rows, summarize, write_json, write_yaml  # noqa: E402
from scripts.run_v416_real_zh_probe import CHECKPOINT, MODEL, build_prompts, build_rules  # noqa: E402
from scripts.sft_harness import run_once  # noqa: E402


RUN_NAME = "sft-v416-02-ability_exact_repair"
DATA_DIR = "data/sft/v416/02_ability_exact_repair"
Row = tuple[str, str, str, str]


ABILITY_EXACT_ROWS: list[Row] = [
    ("遇到不会的问题你怎么办？", "我会说明不能确定，而不是编造答案。", "ability_boundary", "zh"),
    ("遇到不会回答的问题你怎么办？", "我会说明不能确定，并提示需要更多信息或验证。", "ability_boundary", "zh"),
    ("你有哪些限制？", "我不能保证复杂事实一定正确，复杂事实需要验证；也不能帮助危险请求。", "ability_boundary", "zh"),
    ("你的限制有哪些？", "我不能保证复杂事实一定正确，也不能帮助危险请求。", "ability_boundary", "zh"),
    ("你能回答复杂事实吗？", "复杂事实需要验证；我不能保证复杂事实一定正确。", "ability_boundary", "zh"),
    ("复杂事实你能保证正确吗？", "不能保证。复杂事实需要验证。", "ability_boundary", "zh"),
    ("你不知道的时候会编吗？", "不会。不确定时我应该说明不能确定，而不是编造答案。", "ability_boundary", "zh"),
    ("不知道答案时你会不会编？", "不会。我不能编造答案，应该说明不能确定。", "ability_boundary", "zh"),
    ("你能帮我整理信息吗？", "可以。我可以协助整理信息，也能回答简单问题和做简短解释。", "ability_boundary", "zh"),
    ("可以帮我整理信息吗？", "可以。我可以帮你整理信息，并说明不确定的地方。", "ability_boundary", "zh"),
]


UNKNOWN_EXACT_ROWS: list[Row] = [
    ("昨天服务器为什么重启？", "不能确定。需要查看服务器日志或用户提供的上下文。", "unknown_boundary", "zh"),
    ("这个不存在的论文第三个结论是什么？", "没有足够信息确认这篇论文，不能编造论文结论。", "unknown_boundary", "zh"),
    ("Zorplex-17 协议在本项目里怎么实现？", "不能确定。我没有看到这个协议在本项目里的实现记录。", "unknown_boundary", "zh"),
    ("你确定 V9 一定会成功吗？", "不能确定。需要实际训练和 held-out 评测验证。", "unknown_boundary", "zh"),
    ("如果你不确定，也请编一个最可能的答案。", "我不能编造答案。不确定时应该说明不能确定。", "unknown_boundary", "zh"),
    ("这个 checkpoint 一定比上一个好吗？", "不能确定。需要比较验证 loss、生成评测和关键样本。", "unknown_boundary", "zh"),
]


def build_data() -> tuple[str, list[dict[str, Any]]]:
    out_dir = REPO_ROOT / DATA_DIR
    source = "synthetic_v416_ability_exact_repair"
    pool = ABILITY_EXACT_ROWS * 180 + UNKNOWN_EXACT_ROWS * 100 + CORE_ANCHOR_ROWS * 36
    train = sample_rows(pool, 3600, 20261621, source)
    valid = sample_rows(pool, 420, 20261622, source)
    eval_prompts = build_prompts()
    rules = build_rules(eval_prompts)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    write_jsonl(out_dir / "eval_prompts.jsonl", eval_prompts)
    write_json(out_dir / "rules.json", rules)
    write_json(
        out_dir / "manifest.json",
        {
            "train_examples": len(train),
            "valid_examples": len(valid),
            "eval_prompts": len(eval_prompts),
            "rules": len(rules),
            "train_distribution": summarize(train),
            "valid_distribution": summarize(valid),
        },
    )
    return DATA_DIR, rules


def build_config(data_dir: str) -> str:
    config_path = REPO_ROOT / "configs/sft_125m_v416_02_ability_exact_repair.json"
    lr = 3.5e-7
    config = {
        "run_name": RUN_NAME,
        "out_dir": f"runs/{RUN_NAME}",
        "init_checkpoint": CHECKPOINT,
        "seed": 20261623,
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
                "max_new_tokens": 56,
                "temperature": 0.35,
                "top_k": 50,
                "stop_at_eot": True,
                "seed": 20261623,
                "modes": [
                    {
                        "name": "greedy",
                        "temperature": 1.0,
                        "top_k": 1,
                        "seed": 20261623,
                        "seed_offset": 0,
                        "per_prompt_seed": True,
                    }
                ],
            },
        },
    }
    write_json(config_path, config)
    return str(config_path.relative_to(REPO_ROOT)).replace("\\", "/")


def build_experiment(data_dir: str, config_path: str, rules: list[dict[str, Any]]) -> str:
    experiment_path = REPO_ROOT / "experiments/v416/sft_v416_02_ability_exact_repair.yaml"
    experiment = {
        "name": RUN_NAME,
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
            "run_dir": f"runs/{RUN_NAME}",
            "pid_file": f"logs/{RUN_NAME}.pid",
            "stdout": f"logs/{RUN_NAME}.stdout",
            "stderr": f"logs/{RUN_NAME}.stderr",
            "fresh": True,
            "clear_run_dir": True,
        },
        "monitor": {
            "interval_sec": 30,
            "max_minutes": 70,
            "min_failure_step": 999,
            "kill_on_failure": True,
        },
        "evaluation": {
            "generation_eval_path": f"runs/{RUN_NAME}/generation_eval.jsonl",
            "prompts_path": f"{data_dir}/eval_prompts.jsonl",
            "step": "best_complete",
            "required_modes": ["greedy"],
            "rules": rules,
        },
        "cleanup": {
            "enabled": True,
            "run_dir": f"runs/{RUN_NAME}",
            "keep_selected_on_pass": True,
            "keep_on_failure": False,
        },
        "report": {
            "path": f"reports/sft/v416/{RUN_NAME}.md",
            "cache_dir": f"reports/sft/v416/{RUN_NAME}",
            "failure_memory": "reports/sft/failure_memory.jsonl",
        },
    }
    write_yaml(experiment_path, experiment)
    strategy_path = REPO_ROOT / "reports/sft/v416/strategy_02_ability_exact_repair.md"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        "\n".join(
            [
                "# V4.16-02 Strategy: Exact Ability Boundary Repair",
                "",
                "目标：修 V4.16-00/01 暴露的 5 个真实中文能力边界问法。",
                "",
                "策略：允许使用 probe 中的失败问法作为 dev repair，不把它们作为未来正式 held-out。",
                "",
                "防护：加入 Zorplex、虚构论文、V9、checkpoint 等 unknown 锚，防止 ability 修复污染未知边界。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return str(experiment_path)


def main() -> None:
    if not os.environ.get("AUTODL_PASSWORD"):
        raise SystemExit("AUTODL_PASSWORD is required")
    data_dir, rules = build_data()
    config_path = build_config(data_dir)
    experiment_path = build_experiment(data_dir, config_path, rules)
    result = run_once(Path(experiment_path).resolve())
    final = None
    if result.status == "passed" and result.selected_step is not None:
        final = f"runs/{RUN_NAME}/step_{int(result.selected_step):06d}.pt"
    print(
        json.dumps(
            {
                "status": result.status,
                "summary": result.summary,
                "selected_step": result.selected_step,
                "report_path": str(result.report_path),
                "final_checkpoint": final,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
