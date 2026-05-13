from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paramiko
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_sft_outputs import (  # noqa: E402
    append_failure_memory,
    count_jsonl,
    evaluate_rows,
    load_jsonl,
    write_markdown_report,
)


@dataclass
class HarnessResult:
    status: str
    summary: str
    report_path: Path
    selected_step: int | None


def load_experiment(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    data.setdefault("_experiment_path", str(path))
    return data


def q(value: str) -> str:
    return shlex.quote(value)


def remote_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part]
    if not cleaned:
        return "/"
    if parts[0].startswith("/"):
        return "/" + posixpath.join(*cleaned)
    return posixpath.join(*cleaned)


class RemoteSession:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        password_env = cfg.get("password_env")
        password = os.environ.get(str(password_env)) if password_env else cfg.get("password")
        key_filename = cfg.get("key_filename")
        if not password and not key_filename:
            raise RuntimeError("Remote password is missing. Set the configured password_env or key_filename.")
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            hostname=str(cfg["host"]),
            port=int(cfg.get("port", 22)),
            username=str(cfg.get("user", "root")),
            password=password,
            key_filename=key_filename,
            timeout=int(cfg.get("timeout_sec", 20)),
            banner_timeout=int(cfg.get("timeout_sec", 20)),
            auth_timeout=int(cfg.get("timeout_sec", 20)),
        )
        self.sftp = self.ssh.open_sftp()

    def close(self) -> None:
        self.sftp.close()
        self.ssh.close()

    def run(self, command: str, timeout: int = 120) -> tuple[int, str, str]:
        stdin, stdout, stderr = self.ssh.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        return stdout.channel.recv_exit_status(), out, err

    def mkdir_p(self, path: str) -> None:
        parts = []
        current = path
        while current not in ("", "/"):
            parts.append(current)
            current = posixpath.dirname(current)
        for item in reversed(parts):
            try:
                self.sftp.stat(item)
            except FileNotFoundError:
                self.sftp.mkdir(item)

    def upload_file(self, local: Path, remote: str) -> None:
        self.mkdir_p(posixpath.dirname(remote))
        self.sftp.put(str(local), remote)

    def upload_path(self, local: Path, remote: str) -> None:
        if local.is_file():
            self.upload_file(local, remote)
            return
        if not local.is_dir():
            raise FileNotFoundError(local)
        for item in local.rglob("*"):
            if not item.is_file():
                continue
            rel = item.relative_to(local).as_posix()
            self.upload_file(item, remote_join(remote, rel))

    def download_if_exists(self, remote: str, local: Path) -> bool:
        try:
            self.sftp.stat(remote)
        except FileNotFoundError:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote, str(local))
        return True


def run_local_build(cfg: dict[str, Any], root: Path) -> None:
    command = cfg.get("data", {}).get("build_command")
    if not command:
        return
    print(f"[harness] build data: {command}")
    subprocess.run(command, cwd=root, shell=True, check=True)


def upload_inputs(remote: RemoteSession, cfg: dict[str, Any], root: Path) -> None:
    project_dir = str(cfg["remote"]["project_dir"])
    items = cfg.get("upload", {}).get("items", [])
    for item in items:
        local_rel = str(item["local"])
        remote_rel = str(item.get("remote", local_rel)).replace("\\", "/")
        local = root / local_rel
        remote_path = remote_join(project_dir, remote_rel)
        print(f"[harness] upload {local_rel} -> {remote_rel}")
        remote.upload_path(local, remote_path)


def start_training(remote: RemoteSession, cfg: dict[str, Any]) -> None:
    project_dir = str(cfg["remote"]["project_dir"])
    python_bin = str(cfg["remote"].get("python", "/root/miniconda3/bin/python"))
    train_cfg = cfg["train"]
    config_path = str(train_cfg["config"]).replace("\\", "/")
    pid_file = str(train_cfg.get("pid_file", f"logs/{cfg['name']}.pid")).replace("\\", "/")
    stdout = str(train_cfg.get("stdout", f"logs/{cfg['name']}.stdout")).replace("\\", "/")
    stderr = str(train_cfg.get("stderr", f"logs/{cfg['name']}.stderr")).replace("\\", "/")
    fresh = bool(train_cfg.get("fresh", True))
    cleanup = f"rm -f {q(pid_file)} {q(stdout)} {q(stderr)}" if fresh else "true"
    clear_run_dir = "true"
    if bool(train_cfg.get("clear_run_dir", False)):
        run_dir = str(train_cfg.get("run_dir") or posixpath.dirname(str(cfg["evaluation"]["generation_eval_path"]))).replace("\\", "/")
        clear_run_dir = (
            f"mkdir -p {q(run_dir)} && "
            f"find {q(run_dir)} -maxdepth 1 -type f "
            f"\\( -name '*.pt' -o -name 'generation_eval.jsonl' -o -name 'sft_log.jsonl' \\) -delete"
        )
    command = f"""
set -e
cd {q(project_dir)}
mkdir -p {q(posixpath.dirname(pid_file))} {q(posixpath.dirname(stdout))} {q(posixpath.dirname(stderr))}
{cleanup}
{clear_run_dir}
nohup {q(python_bin)} -u -m gpt_small.training.sft --config {q(config_path)} > {q(stdout)} 2> {q(stderr)} &
echo $! > {q(pid_file)}
echo started_pid=$(cat {q(pid_file)})
"""
    rc, out, err = remote.run(command, timeout=120)
    print(out.strip())
    if rc != 0:
        raise RuntimeError(f"failed to start training: {err}")


def remote_process_running(remote: RemoteSession, cfg: dict[str, Any]) -> bool:
    project_dir = str(cfg["remote"]["project_dir"])
    pid_file = str(cfg["train"].get("pid_file", f"logs/{cfg['name']}.pid")).replace("\\", "/")
    command = f"""
cd {q(project_dir)}
pid=$(cat {q(pid_file)} 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  echo running
else
  echo stopped
fi
"""
    rc, out, _err = remote.run(command, timeout=30)
    return rc == 0 and "running" in out


def kill_training(remote: RemoteSession, cfg: dict[str, Any]) -> None:
    project_dir = str(cfg["remote"]["project_dir"])
    pid_file = str(cfg["train"].get("pid_file", f"logs/{cfg['name']}.pid")).replace("\\", "/")
    command = f"""
cd {q(project_dir)}
pid=$(cat {q(pid_file)} 2>/dev/null || true)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
  kill "$pid" || true
  sleep 3
  if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid" || true; fi
fi
"""
    remote.run(command, timeout=60)


def download_artifacts(remote: RemoteSession, cfg: dict[str, Any], root: Path, cache_dir: Path) -> dict[str, Path]:
    project_dir = str(cfg["remote"]["project_dir"])
    evaluation = cfg["evaluation"]
    remote_generation = str(evaluation["generation_eval_path"]).replace("\\", "/")
    remote_log = str(cfg.get("train", {}).get("sft_log_path", "") or "").replace("\\", "/")
    if not remote_log:
        out_dir = posixpath.dirname(remote_generation)
        remote_log = posixpath.join(out_dir, "sft_log.jsonl")

    local_generation = cache_dir / "generation_eval.jsonl"
    local_log = cache_dir / "sft_log.jsonl"
    paths: dict[str, Path] = {}
    if remote.download_if_exists(remote_join(project_dir, remote_generation), local_generation):
        paths["generation"] = local_generation
    if remote.download_if_exists(remote_join(project_dir, remote_log), local_log):
        paths["log"] = local_log

    prompts_value = evaluation.get("prompts_path")
    if prompts_value:
        prompts_path = Path(str(prompts_value))
        if not prompts_path.is_absolute():
            prompts_path = root / prompts_path
        paths["prompts"] = prompts_path
    return paths


def evaluate_current(cfg: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    if "generation" not in paths:
        return {
            "status": "incomplete",
            "selected_step": None,
            "summary": "generation_eval.jsonl has not been created yet.",
            "rules": [],
            "hard_failed": [],
            "soft_failed": [],
        }
    evaluation = cfg["evaluation"]
    expected_prompts = count_jsonl(paths.get("prompts"))
    return evaluate_rows(
        load_jsonl(paths["generation"]),
        list(evaluation.get("rules", [])),
        expected_prompts=expected_prompts,
        required_modes=[str(item) for item in evaluation.get("required_modes", ["greedy"])],
        step=str(evaluation.get("step", "latest_complete")),
    )


def log_tail(path: Path, max_lines: int = 8) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]


def finish_report(
    cfg: dict[str, Any],
    result: dict[str, Any],
    report_path: Path,
    failure_memory: Path,
    status_override: str | None = None,
    extra: dict[str, Any] | None = None,
) -> HarnessResult:
    if status_override:
        result = dict(result)
        result["status"] = status_override
    write_markdown_report(report_path, str(cfg["name"]), result, extra=extra)
    append_failure_memory(failure_memory, str(cfg["name"]), result)
    return HarnessResult(
        status=str(result.get("status")),
        summary=str(result.get("summary")),
        report_path=report_path,
        selected_step=result.get("selected_step"),
    )


def cleanup_checkpoints(remote: RemoteSession, cfg: dict[str, Any], result: dict[str, Any]) -> str:
    cleanup = cfg.get("cleanup", {})
    if not cleanup.get("enabled", False):
        return "disabled"

    status = str(result.get("status"))
    selected_step = result.get("selected_step")
    keep: set[str] = {str(item) for item in cleanup.get("keep_files", [])}
    keep_selected_on_pass = bool(cleanup.get("keep_selected_on_pass", True))
    keep_on_failure = bool(cleanup.get("keep_on_failure", False))
    if status == "passed" and keep_selected_on_pass and selected_step is not None:
        keep.add(f"step_{int(selected_step):06d}.pt")
    if status != "passed" and not keep_on_failure:
        keep = {name for name in keep if name == "latest.pt"}

    project_dir = str(cfg["remote"]["project_dir"])
    run_dir = str(cleanup.get("run_dir") or cfg["train"].get("run_dir") or posixpath.dirname(str(cfg["evaluation"]["generation_eval_path"]))).replace("\\", "/")
    keep_args = " ".join(q(item) for item in sorted(keep))
    command = f"""
set -e
cd {q(project_dir)}
run_dir={q(run_dir)}
keep_list="{keep_args}"
if [ -d "$run_dir" ]; then
  for file in "$run_dir"/*.pt; do
    [ -e "$file" ] || continue
    base=$(basename "$file")
    keep=no
    for item in $keep_list; do
      if [ "$base" = "$item" ]; then keep=yes; fi
    done
    if [ "$keep" = no ]; then
      rm -f "$file"
      echo deleted:$file
    else
      echo kept:$file
    fi
  done
fi
"""
    rc, out, err = remote.run(command, timeout=120)
    if rc != 0:
        return f"cleanup_failed: {err.strip()}"
    return out.strip() or "no_checkpoints"


def run_once(experiment_path: Path) -> HarnessResult:
    cfg = load_experiment(experiment_path)
    root = Path(cfg.get("local_root", REPO_ROOT)).resolve()
    name = str(cfg["name"])
    report_cfg = cfg.get("report", {})
    report_path = root / str(report_cfg.get("path", f"reports/sft/{name}_report.md"))
    failure_memory = root / str(report_cfg.get("failure_memory", "reports/sft/failure_memory.jsonl"))
    cache_dir = root / str(report_cfg.get("cache_dir", f"reports/sft/{name}_artifacts"))

    run_local_build(cfg, root)
    remote = RemoteSession(cfg["remote"])
    try:
        upload_inputs(remote, cfg, root)
        start_training(remote, cfg)
        monitor = cfg.get("monitor", {})
        interval_sec = int(monitor.get("interval_sec", 120))
        max_minutes = float(monitor.get("max_minutes", 120))
        min_failure_step = int(monitor.get("min_failure_step", 0))
        kill_on_failure = bool(monitor.get("kill_on_failure", True))
        started = time.time()
        latest_result: dict[str, Any] = {
            "status": "incomplete",
            "selected_step": None,
            "summary": "No evaluation has completed yet.",
            "rules": [],
            "hard_failed": [],
            "soft_failed": [],
        }

        while True:
            time.sleep(interval_sec)
            running = remote_process_running(remote, cfg)
            paths = download_artifacts(remote, cfg, root, cache_dir)
            latest_result = evaluate_current(cfg, paths)
            print(f"[harness] {name}: running={running} {latest_result.get('summary')}")

            selected_step = latest_result.get("selected_step")
            complete_steps = latest_result.get("complete_steps") or []
            latest_complete_step = max([int(item) for item in complete_steps], default=-1)
            can_fail = latest_complete_step >= min_failure_step
            if latest_result.get("status") == "failed" and can_fail:
                if kill_on_failure and running:
                    kill_training(remote, cfg)
                cleanup_note = cleanup_checkpoints(remote, cfg, latest_result)
                return finish_report(
                    cfg,
                    latest_result,
                    report_path,
                    failure_memory,
                    status_override="failed",
                    extra={"early_stop": True, "reason": "hard gate failed", "cleanup": cleanup_note},
                )

            if not running:
                cleanup_note = cleanup_checkpoints(remote, cfg, latest_result)
                return finish_report(
                    cfg,
                    latest_result,
                    report_path,
                    failure_memory,
                    extra={"process": "stopped", "cleanup": cleanup_note},
                )

            if time.time() - started > max_minutes * 60:
                if kill_on_failure:
                    kill_training(remote, cfg)
                timeout_result = dict(latest_result)
                timeout_result["status"] = "failed"
                timeout_result["summary"] = f"Timed out after {max_minutes} minutes."
                cleanup_note = cleanup_checkpoints(remote, cfg, timeout_result)
                return finish_report(
                    cfg,
                    timeout_result,
                    report_path,
                    failure_memory,
                    extra={"timeout_minutes": max_minutes, "cleanup": cleanup_note},
                )
    finally:
        remote.close()


def run_chain(experiment_path: Path, depth: int = 1, max_chain: int | None = None) -> HarnessResult:
    result = run_once(experiment_path)
    cfg = load_experiment(experiment_path)
    iteration = cfg.get("iteration", {})
    configured_max = int(iteration.get("max_chain", 1))
    max_chain = configured_max if max_chain is None else min(max_chain, configured_max)
    next_experiment = iteration.get("next_experiment")
    if (
        result.status == "passed"
        and bool(iteration.get("continue_on_pass", False))
        and next_experiment
        and depth < max_chain
    ):
        next_path = Path(str(next_experiment))
        if not next_path.is_absolute():
            next_path = experiment_path.parent / next_path
        print(f"[harness] continuing to next experiment: {next_path}")
        return run_chain(next_path.resolve(), depth=depth + 1, max_chain=max_chain)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    result = run_chain(Path(args.experiment).resolve())
    print(json.dumps(result.__dict__ | {"report_path": str(result.report_path)}, ensure_ascii=False, indent=2))
    if result.status == "failed":
        sys.exit(2)
    if result.status == "incomplete":
        sys.exit(1)


if __name__ == "__main__":
    main()
