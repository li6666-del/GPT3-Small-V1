from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.sft_data import IGNORE_INDEX, SFTJsonlDataset, TextTokenizer
from gpt_small.training.train import build_optimizer, save_checkpoint
from gpt_small.training.utils import (
    cosine_lr,
    load_json,
    resolve_device,
    resolve_dtype,
    set_seed,
    write_jsonl,
)


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: SFTJsonlDataset,
    valid_data: SFTJsonlDataset,
    batch_size: int,
    eval_iters: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, dataset in (("train", train_data), ("valid", valid_data)):
        losses = torch.empty(eval_iters)
        label_tokens = 0
        for k in range(eval_iters):
            x, y = dataset.get_batch(batch_size)
            label_tokens += int((y != IGNORE_INDEX).sum().item())
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[f"{split}_loss"] = losses.mean().item()
        out[f"{split}_label_tokens"] = float(label_tokens)
    model.train()
    return out


def build_tokenizer(config: dict) -> TextTokenizer | None:
    data_cfg = config["data"]
    if data_cfg.get("ids_only", False):
        return None
    return TextTokenizer(
        tokenizer_json_path=data_cfg.get("tokenizer_json_path"),
        vocab_path=data_cfg.get("vocab_path"),
        merges_path=data_cfg.get("merges_path"),
    )


def load_model_weights(
    model: TransformerLM,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def load_generation_prompts(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(prompt_path)
    prompts: list[dict[str, Any]] = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                raise ValueError(f"{prompt_path}:{line_no} is missing a prompt")
            prompts.append(
                {
                    "id": str(row.get("id", f"prompt_{line_no:03d}")),
                    "prompt": prompt,
                    "category": row.get("category"),
                    "language": row.get("language"),
                }
            )
    return prompts


@torch.no_grad()
def run_generation_eval(
    model: TransformerLM,
    tokenizer: TextTokenizer | None,
    prompts: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
    step: int,
    device: torch.device,
) -> None:
    if tokenizer is None or not prompts:
        return

    max_new_tokens = int(config.get("max_new_tokens", 96))
    temperature = float(config.get("temperature", 0.35))
    top_k = config.get("top_k", 50)
    top_k = int(top_k) if top_k is not None else None
    base_seed = int(config.get("seed", 20260512))
    default_modes: list[dict[str, Any]] = [
        {
            "name": "sample_sequence",
            "temperature": temperature,
            "top_k": top_k,
            "seed": base_seed + step,
            "per_prompt_seed": False,
        }
    ]
    modes = config.get("modes", default_modes)

    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None

    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("a", encoding="utf-8", newline="\n") as f:
            for mode_index, mode in enumerate(modes):
                mode_name = str(mode.get("name", f"mode_{mode_index}"))
                mode_temperature = float(mode.get("temperature", temperature))
                mode_top_k = mode.get("top_k", top_k)
                mode_top_k = int(mode_top_k) if mode_top_k is not None else None
                mode_seed = int(mode.get("seed", base_seed)) + int(mode.get("seed_offset", step))
                per_prompt_seed = bool(mode.get("per_prompt_seed", False))

                if not per_prompt_seed:
                    torch.manual_seed(mode_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(mode_seed)

                for prompt_index, item in enumerate(prompts):
                    seed = mode_seed + prompt_index if per_prompt_seed else mode_seed
                    if per_prompt_seed:
                        torch.manual_seed(seed)
                        if device.type == "cuda":
                            torch.cuda.manual_seed_all(seed)

                    formatted_prompt = f"User: {item['prompt']}\nAssistant: "
                    input_ids = tokenizer.encode(formatted_prompt)
                    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
                    output = model.generate(
                        ids,
                        max_new_tokens=max_new_tokens,
                        temperature=mode_temperature,
                        top_k=mode_top_k,
                    )
                    new_ids = output[0].tolist()[len(input_ids) :]
                    if config.get("stop_at_eot", True) and tokenizer.eot_id in new_ids:
                        new_ids = new_ids[: new_ids.index(tokenizer.eot_id)]
                    row = {
                        "step": step,
                        "mode": mode_name,
                        "seed": seed,
                        "temperature": mode_temperature,
                        "top_k": mode_top_k,
                        "id": item["id"],
                        "category": item.get("category"),
                        "language": item.get("language"),
                        "prompt": item["prompt"],
                        "output": tokenizer.decode(new_ids).strip(),
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(config["seed"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "sft_log.jsonl"

    device = resolve_device(config["device"])
    amp_dtype = resolve_dtype(config["dtype"], device)
    model_config = GPTConfig(**config["model"])
    model = TransformerLM(model_config).to(device)
    optimizer = build_optimizer(model, config["optim"])

    step = 0
    best_valid_loss = float("inf")
    latest_path = out_dir / "latest.pt"
    resume = config["train"].get("resume", False)
    if resume and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"]) + 1
        best_valid_loss = float(checkpoint["best_valid_loss"])
        write_jsonl(log_path, {"event": "resume", "checkpoint": str(latest_path), "step": step})
    elif config.get("init_checkpoint"):
        load_model_weights(model, config["init_checkpoint"], device)
        write_jsonl(
            log_path,
            {"event": "init_from_checkpoint", "checkpoint": str(config["init_checkpoint"])},
        )

    if config.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    tokenizer = build_tokenizer(config)
    data_cfg = config["data"]
    train_data = SFTJsonlDataset(
        data_cfg["train_path"],
        model_config.context_length,
        tokenizer=tokenizer,
        pad_token_id=data_cfg.get("pad_token_id"),
        train_eot=data_cfg.get("train_eot", True),
        truncate=data_cfg.get("truncate", True),
        device=device,
    )
    valid_data = SFTJsonlDataset(
        data_cfg["valid_path"],
        model_config.context_length,
        tokenizer=tokenizer,
        pad_token_id=data_cfg.get("pad_token_id"),
        train_eot=data_cfg.get("train_eot", True),
        truncate=data_cfg.get("truncate", True),
        device=device,
    )
    write_jsonl(
        log_path,
        {
            "event": "data_loaded",
            "train_examples": len(train_data),
            "valid_examples": len(valid_data),
        },
    )
    generation_eval_cfg = config["train"].get("generation_eval", {})
    generation_prompts = load_generation_prompts(generation_eval_cfg.get("prompts_path")) if generation_eval_cfg.get("enabled", False) else []
    generation_eval_path = out_dir / generation_eval_cfg.get("output_path", "generation_eval.jsonl")
    if generation_prompts:
        write_jsonl(
            log_path,
            {
                "event": "generation_eval_loaded",
                "prompts": len(generation_prompts),
                "output_path": str(generation_eval_path),
            },
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    model.train()
    start_time = time.time()
    max_steps = config["train"]["max_steps"]
    grad_accum = config["train"]["gradient_accumulation_steps"]
    batch_size = data_cfg["batch_size"]
    total_input_tokens = 0
    total_label_tokens = 0

    while step < max_steps:
        lr = cosine_lr(
            step,
            max_steps,
            config["train"]["warmup_steps"],
            config["optim"]["learning_rate"],
            config["optim"]["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_input_tokens = 0
        step_label_tokens = 0
        for _ in range(grad_accum):
            x, y = train_data.get_batch(batch_size)
            step_input_tokens += x.numel()
            step_label_tokens += int((y != IGNORE_INDEX).sum().item())
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        total_input_tokens += step_input_tokens
        total_label_tokens += step_label_tokens

        if config["optim"]["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optim"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % config["train"]["log_interval"] == 0:
            row = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "elapsed_sec": time.time() - start_time,
                "input_tokens": total_input_tokens,
                "label_tokens": total_label_tokens,
                "step_input_tokens": step_input_tokens,
                "step_label_tokens": step_label_tokens,
            }
            print(row)
            write_jsonl(log_path, row)

        if step % config["train"]["eval_interval"] == 0 or step == max_steps - 1:
            losses = estimate_loss(
                model,
                train_data,
                valid_data,
                batch_size,
                config["train"]["eval_iters"],
                amp_dtype,
                device,
            )
            best_valid_loss = min(best_valid_loss, losses["valid_loss"])
            write_jsonl(
                log_path,
                {
                    "step": step,
                    "eval_train_loss": losses["train_loss"],
                    "eval_valid_loss": losses["valid_loss"],
                    "best_valid_loss": best_valid_loss,
                    "eval_train_label_tokens": losses["train_label_tokens"],
                    "eval_valid_label_tokens": losses["valid_label_tokens"],
                },
            )
            if generation_prompts and (
                step % int(generation_eval_cfg.get("interval", config["train"]["eval_interval"])) == 0
                or step == max_steps - 1
            ):
                target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                run_generation_eval(
                    target_model,
                    tokenizer,
                    generation_prompts,
                    generation_eval_cfg,
                    generation_eval_path,
                    step,
                    device,
                )

        if step % config["train"]["save_interval"] == 0 or step == max_steps - 1:
            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(latest_path, target_model, optimizer, config, step, best_valid_loss)
            save_checkpoint(out_dir / f"step_{step:06d}.pt", target_model, optimizer, config, step, best_valid_loss)

        step += 1


if __name__ == "__main__":
    main()
