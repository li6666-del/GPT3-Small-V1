from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime for clean CLI errors.
    yaml = None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL") from exc
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install requirements.txt first.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def count_jsonl(path: Path | None) -> int | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def row_identity(row: dict[str, Any], fallback: int) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    if row.get("prompt") is not None:
        return str(row["prompt"])
    return f"row_{fallback}"


def dedupe_generation_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    latest_by_key: dict[tuple[int, str, str], dict[str, Any]] = {}
    order: list[tuple[int, str, str]] = []
    duplicate_count = 0
    for index, row in enumerate(rows):
        if row.get("step") is None or row.get("mode") is None:
            order.append((index, "", f"malformed_{index}"))
            latest_by_key[(index, "", f"malformed_{index}")] = row
            continue
        key = (int(row["step"]), str(row["mode"]), row_identity(row, index))
        if key in latest_by_key:
            duplicate_count += 1
        else:
            order.append(key)
        latest_by_key[key] = row
    return [latest_by_key[key] for key in order], duplicate_count


def output_text(row: dict[str, Any]) -> str:
    for key in ("output", "generated", "text", "completion", "response"):
        value = row.get(key)
        if value is not None:
            return str(value).strip()
    return ""


def enrich_rows_with_prompts(rows: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not prompts:
        return rows
    by_id = {str(row.get("id")): row for row in prompts if row.get("id") is not None}
    by_prompt = {str(row.get("prompt")): row for row in prompts if row.get("prompt") is not None}
    enriched: list[dict[str, Any]] = []
    for row in rows:
        prompt_row = by_id.get(str(row.get("id"))) or by_prompt.get(str(row.get("prompt")))
        if not prompt_row:
            enriched.append(row)
            continue
        merged = dict(row)
        for key in ("expected", "eval_set"):
            if key not in merged and key in prompt_row:
                merged[key] = prompt_row[key]
        enriched.append(merged)
    return enriched


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def contains_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def row_matches(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    if rule.get("id") is not None and str(row.get("id")) != str(rule["id"]):
        return False
    if rule.get("prompt") is not None and str(row.get("prompt")) != str(rule["prompt"]):
        return False
    for needle in as_list(rule.get("prompt_contains")):
        if needle not in str(row.get("prompt", "")):
            return False
    if rule.get("category") is not None and str(row.get("category")) != str(rule["category"]):
        return False
    for needle in as_list(rule.get("category_contains")):
        if needle not in str(row.get("category", "")):
            return False
    if rule.get("language") is not None and str(row.get("language")) != str(rule["language"]):
        return False
    return True


def check_output(text: str, rule: dict[str, Any], row: dict[str, Any] | None = None) -> list[str]:
    reasons: list[str] = []
    expected = rule.get("equals")
    if expected is not None and text != str(expected):
        reasons.append(f"expected exact output {expected!r}")

    if rule.get("equals_expected", False):
        if row is None or row.get("expected") is None:
            reasons.append("rule requires expected field, but row has no expected value")
        else:
            expected_value = str(row.get("expected")).strip()
            if text != expected_value:
                reasons.append(f"expected exact output from row expected field {expected_value!r}")

    must_include_all = as_list(rule.get("must_include_all")) + as_list(rule.get("must_include"))
    missing = [needle for needle in must_include_all if needle not in text]
    if missing:
        reasons.append("missing required text: " + ", ".join(repr(item) for item in missing))

    must_include_any = as_list(rule.get("must_include_any"))
    if must_include_any and not contains_any(text, must_include_any):
        reasons.append("missing any of: " + ", ".join(repr(item) for item in must_include_any))

    forbidden = as_list(rule.get("must_not_include_any")) + as_list(rule.get("must_not_include"))
    hit_forbidden = [needle for needle in forbidden if needle in text]
    if hit_forbidden:
        reasons.append("contains forbidden text: " + ", ".join(repr(item) for item in hit_forbidden))

    if rule.get("non_empty", False) and not text:
        reasons.append("output is empty")

    max_chars = rule.get("max_chars")
    if max_chars is not None and len(text) > int(max_chars):
        reasons.append(f"output length {len(text)} exceeds max_chars={max_chars}")
    return reasons


def rule_layer(rule: dict[str, Any]) -> str:
    layer = str(rule.get("layer") or rule.get("severity", "stage")).lower()
    if layer == "hard":
        return "stage"
    if layer == "soft":
        return "observe"
    if layer not in {"main", "stage", "observe"}:
        return "stage"
    return layer


def is_blocking_rule(rule: dict[str, Any]) -> bool:
    return rule_layer(rule) in {"main", "stage"}


def complete_steps(
    rows: list[dict[str, Any]],
    expected_prompts: int | None,
    required_modes: list[str],
) -> list[int]:
    if not required_modes:
        required_modes = sorted({str(row.get("mode")) for row in rows if row.get("mode") is not None})
    prompt_ids: dict[tuple[int, str], set[str]] = {}
    for index, row in enumerate(rows):
        if row.get("step") is None or row.get("mode") is None:
            continue
        key = (int(row["step"]), str(row["mode"]))
        prompt_ids.setdefault(key, set()).add(row_identity(row, index))
    steps = sorted({step for step, _mode in prompt_ids})
    complete: list[int] = []
    for step in steps:
        mode_counts = [len(prompt_ids.get((step, mode), set())) for mode in required_modes]
        if expected_prompts is None:
            if all(count > 0 for count in mode_counts):
                complete.append(step)
        elif all(count >= expected_prompts for count in mode_counts):
            complete.append(step)
    return complete


def select_step(
    rows: list[dict[str, Any]],
    expected_prompts: int | None,
    required_modes: list[str],
    step: str | int = "latest_complete",
) -> int | None:
    if not rows:
        return None
    if isinstance(step, int):
        return step
    if str(step).isdigit():
        return int(step)
    steps = sorted({int(row["step"]) for row in rows if row.get("step") is not None})
    if not steps:
        return None
    if step == "latest":
        return steps[-1]
    complete = complete_steps(rows, expected_prompts, required_modes)
    if complete:
        return complete[-1]
    return None


def evaluate_at_step(rows: list[dict[str, Any]], rules: list[dict[str, Any]], selected_step: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "incomplete",
        "selected_step": selected_step,
        "rules": [],
        "main_failed": [],
        "stage_failed": [],
        "observe_failed": [],
        "hard_failed": [],
        "soft_failed": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    step_rows = [row for row in rows if int(row.get("step", -1)) == selected_step]
    for rule in rules:
        mode = str(rule.get("mode", "greedy"))
        layer = rule_layer(rule)
        severity = str(rule.get("severity", layer))
        matched = [row for row in step_rows if str(row.get("mode")) == mode and row_matches(row, rule)]
        passed_rows = []
        failed_rows = []
        for row in matched:
            text = output_text(row)
            reasons = check_output(text, rule, row)
            sample = {
                "id": row.get("id"),
                "category": row.get("category"),
                "prompt": row.get("prompt"),
                "output": text,
                "reasons": reasons,
            }
            if reasons:
                failed_rows.append(sample)
            else:
                passed_rows.append(sample)

        min_matches = int(rule.get("min_matches", 1))
        min_pass_ratio = float(rule.get("min_pass_ratio", 1.0))
        matched_count = len(matched)
        passed_count = len(passed_rows)
        pass_ratio = passed_count / matched_count if matched_count else 0.0
        ok = matched_count >= min_matches and pass_ratio >= min_pass_ratio
        if matched_count < min_matches:
            failed_rows.append(
                {
                    "id": None,
                    "category": rule.get("category") or rule.get("category_contains"),
                    "prompt": rule.get("prompt"),
                    "output": "",
                    "reasons": [f"matched {matched_count} rows, need at least {min_matches}"],
                }
            )

        rule_result = {
            "name": str(rule.get("name", "unnamed_rule")),
            "severity": severity,
            "layer": layer,
            "mode": mode,
            "ok": ok,
            "matched": matched_count,
            "passed": passed_count,
            "pass_ratio": pass_ratio,
            "failed_samples": failed_rows[: int(rule.get("max_failed_samples", 5))],
        }
        result["rules"].append(rule_result)
        if not ok and layer == "main":
            result["main_failed"].append(rule_result)
            result["hard_failed"].append(rule_result)
        elif not ok and layer == "stage":
            result["stage_failed"].append(rule_result)
            result["hard_failed"].append(rule_result)
        elif not ok:
            result["observe_failed"].append(rule_result)
            result["soft_failed"].append(rule_result)

    result["status"] = "failed" if result["hard_failed"] else "passed"
    result["summary"] = summarize_result(result)
    return result


def step_score(result: dict[str, Any]) -> tuple[int, int, int, int, int, int]:
    selected_step = int(result.get("selected_step") or 0)
    blocking_rules = [
        rule
        for rule in result.get("rules", [])
        if rule_layer(rule) in {"main", "stage"}
    ]
    blocking_ratio_score = int(sum(float(rule.get("pass_ratio", 0.0)) * 1000 for rule in blocking_rules))
    blocking_passed = int(sum(int(rule.get("passed", 0)) for rule in blocking_rules))
    return (
        len(result.get("main_failed", [])),
        len(result.get("stage_failed", [])),
        len(result.get("observe_failed", [])),
        -blocking_ratio_score,
        -blocking_passed,
        -selected_step,
    )


def evaluate_rows(
    rows: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    expected_prompts: int | None = None,
    required_modes: list[str] | None = None,
    step: str | int = "latest_complete",
) -> dict[str, Any]:
    rows, duplicate_count = dedupe_generation_rows(rows)
    required_modes = required_modes or []
    complete = complete_steps(rows, expected_prompts, required_modes)
    selected_step = select_step(rows, expected_prompts, required_modes, step)
    if step == "best_complete":
        candidates = [evaluate_at_step(rows, rules, item) for item in complete]
        if not candidates:
            selected_step = None
        else:
            result = min(candidates, key=step_score)
            result["candidate_scores"] = [
                {
                    "step": item["selected_step"],
                    "main_failed": len(item.get("main_failed", [])),
                    "stage_failed": len(item.get("stage_failed", [])),
                    "observe_failed": len(item.get("observe_failed", [])),
                    "status": item["status"],
                }
                for item in sorted(candidates, key=lambda row: int(row["selected_step"]))
            ]
            result["expected_prompts"] = expected_prompts
            result["required_modes"] = required_modes
            result["complete_steps"] = complete
            result["duplicate_rows"] = duplicate_count
            return result

    if selected_step is None:
        return {
            "status": "incomplete",
            "selected_step": selected_step,
            "expected_prompts": expected_prompts,
            "required_modes": required_modes,
            "rules": [],
            "main_failed": [],
            "stage_failed": [],
            "observe_failed": [],
            "hard_failed": [],
            "soft_failed": [],
            "complete_steps": complete,
            "duplicate_rows": duplicate_count,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": "No complete generation eval step is available yet.",
        }

    result = evaluate_at_step(rows, rules, int(selected_step))
    result["expected_prompts"] = expected_prompts
    result["required_modes"] = required_modes
    result["complete_steps"] = complete
    result["duplicate_rows"] = duplicate_count
    result["summary"] = summarize_result(result)
    return result


def summarize_result(result: dict[str, Any]) -> str:
    step = result.get("selected_step")
    if result.get("status") == "incomplete":
        return "Generation eval is incomplete."
    main = result.get("main_failed") or []
    stage = result.get("stage_failed") or []
    observe = result.get("observe_failed") or []
    if main:
        names = ", ".join(rule["name"] for rule in main)
        return f"Step {step} failed main gates: {names}."
    if stage:
        names = ", ".join(rule["name"] for rule in stage)
        return f"Step {step} failed stage gates: {names}."
    if observe:
        names = ", ".join(rule["name"] for rule in observe)
        return f"Step {step} passed hard gates, with soft warnings: {names}."
    return f"Step {step} passed all configured gates."


def advice_for_rule(rule_name: str) -> str:
    lower = rule_name.lower()
    if "identity" in lower:
        return "身份锚被侵蚀。下一轮应增加 identity anchor，并降低 unknown/ability 的相对权重。"
    if "unknown" in lower or "fabrication" in lower or "certainty" in lower:
        return "未知边界不足。下一轮应增加未知陷阱样本，但必须配套身份锚和 stop 锚。"
    if "ability" in lower:
        return "能力描述不稳。下一轮应把 ability 回答限制为短句，避免混入训练流程描述。"
    if "stop" in lower:
        return "停止锚退化。下一轮应增加 exact stop 样本，并降低 max_new_tokens 干扰。"
    if "refusal" in lower:
        return "拒绝边界退化。下一轮应增加安全拒绝样本，并检查是否被普通帮助样本冲淡。"
    return "需要检查失败样本，避免只按 loss 决策。"


def write_markdown_report(
    path: Path,
    experiment_name: str,
    result: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    lines = [
        f"# {experiment_name} SFT Harness Report",
        "",
        f"- status: `{result.get('status')}`",
        f"- selected_step: `{result.get('selected_step')}`",
        f"- summary: {result.get('summary')}",
    ]
    if result.get("expected_prompts") is not None:
        lines.append(f"- expected_prompts: `{result.get('expected_prompts')}`")
    if result.get("duplicate_rows"):
        lines.append(f"- duplicate_generation_rows: `{result.get('duplicate_rows')}`")
    for key, value in extra.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(["", "## Gate Results", ""])
    for rule in result.get("rules", []):
        mark = "PASS" if rule.get("ok") else "FAIL"
        lines.append(
            f"- `{mark}` {rule['name']} [{rule.get('layer', rule['severity'])}/{rule['mode']}]: "
            f"{rule['passed']}/{rule['matched']} ({rule['pass_ratio']:.2f})"
        )

    if result.get("candidate_scores"):
        lines.extend(["", "## Best-Step Candidates", ""])
        for item in result["candidate_scores"]:
            lines.append(
                f"- step `{item['step']}`: main={item['main_failed']}, "
                f"stage={item['stage_failed']}, observe={item['observe_failed']}, status={item['status']}"
            )

    failed = result.get("hard_failed", []) + result.get("soft_failed", [])
    if failed:
        lines.extend(["", "## Failed Samples", ""])
        for rule in failed:
            lines.append(f"### {rule['name']}")
            lines.append("")
            for sample in rule.get("failed_samples", []):
                lines.append(f"- prompt: {sample.get('prompt')}")
                lines.append(f"  output: {sample.get('output')}")
                lines.append(f"  reason: {'; '.join(sample.get('reasons', []))}")
            lines.append("")
        lines.extend(["## Avoid Next Time", ""])
        seen: set[str] = set()
        for rule in failed:
            advice = advice_for_rule(str(rule["name"]))
            if advice not in seen:
                lines.append(f"- {advice}")
                seen.add(advice)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def append_failure_memory(path: Path, experiment_name: str, result: dict[str, Any]) -> None:
    if result.get("status") != "failed":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "step": result.get("selected_step"),
        "summary": result.get("summary"),
        "failed_rules": [
            {
                "name": rule["name"],
                "mode": rule["mode"],
                "matched": rule["matched"],
                "passed": rule["passed"],
                "advice": advice_for_rule(str(rule["name"])),
                "samples": rule.get("failed_samples", []),
            }
            for rule in result.get("hard_failed", [])
        ],
    }
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rules_from_experiment(experiment: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    evaluation = experiment.get("evaluation", {})
    return list(evaluation.get("rules", [])), [str(item) for item in evaluation.get("required_modes", ["greedy"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", help="Experiment YAML containing evaluation.rules")
    parser.add_argument("--generation", help="generation_eval.jsonl path")
    parser.add_argument("--prompts", help="eval_prompts.jsonl path, used to detect complete steps")
    parser.add_argument("--step", default="latest_complete")
    parser.add_argument("--report-md")
    parser.add_argument("--report-json")
    parser.add_argument("--failure-memory")
    args = parser.parse_args()

    experiment: dict[str, Any] = {}
    root = Path.cwd()
    if args.experiment:
        exp_path = Path(args.experiment)
        experiment = load_yaml(exp_path)
        root = Path(experiment.get("local_root", exp_path.parent)).resolve()

    evaluation = experiment.get("evaluation", {})
    generation_path_value = args.generation or evaluation.get("local_generation_eval_path") or evaluation.get("generation_eval_path")
    if not generation_path_value:
        raise SystemExit("--generation or evaluation.generation_eval_path is required")
    generation_path = Path(generation_path_value)
    prompts_path_value = args.prompts or evaluation.get("prompts_path")
    if not generation_path.is_absolute():
        generation_path = root / generation_path
    prompts_path = Path(prompts_path_value) if prompts_path_value else None
    if prompts_path is not None and not prompts_path.is_absolute():
        prompts_path = root / prompts_path

    if not generation_path.exists() or not generation_path.is_file():
        raise SystemExit(f"generation JSONL not found or not a file: {generation_path}")
    rules, required_modes = rules_from_experiment(experiment)
    if not rules:
        raise SystemExit("No evaluation rules configured")

    expected_prompts = count_jsonl(prompts_path)
    result = evaluate_rows(
        enrich_rows_with_prompts(load_jsonl(generation_path), load_jsonl(prompts_path) if prompts_path else []),
        rules,
        expected_prompts=expected_prompts,
        required_modes=required_modes,
        step=args.step,
    )

    experiment_name = str(experiment.get("name", generation_path.parent.name))
    if args.report_md:
        write_markdown_report(Path(args.report_md), experiment_name, result)
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.failure_memory:
        append_failure_memory(Path(args.failure_memory), experiment_name, result)

    print(json.dumps({"status": result["status"], "summary": result["summary"]}, ensure_ascii=False))
    if result["status"] == "failed":
        sys.exit(2)
    if result["status"] == "incomplete":
        sys.exit(1)


if __name__ == "__main__":
    main()
