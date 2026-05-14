from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from gpt_small.data import MemmapTokenDataset
from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.training.utils import (
    cosine_lr,
    load_json,
    resolve_device,
    resolve_dtype,
    set_seed,
    write_jsonl,
)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg["learning_rate"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    train_data: MemmapTokenDataset,
    valid_data: MemmapTokenDataset,
    batch_size: int,
    eval_iters: int,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, dataset in (("train", train_data), ("valid", valid_data)):
        losses = torch.empty(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(batch_size)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    config: dict,
    step: int,
    best_valid_loss: float,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "best_valid_loss": best_valid_loss,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(config["seed"])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    device = resolve_device(config["device"])
    amp_dtype = resolve_dtype(config["dtype"], device)
    model_config = GPTConfig(**config["model"])
    model = TransformerLM(model_config).to(device)
    optimizer = build_optimizer(model, config["optim"])
    if config.get("compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_data = MemmapTokenDataset(
        config["data"]["train_path"],
        model_config.context_length,
        config["data"]["dtype"],
        device,
    )
    valid_data = MemmapTokenDataset(
        config["data"]["valid_path"],
        model_config.context_length,
        config["data"]["dtype"],
        device,
    )

    step = 0
    best_valid_loss = float("inf")
    latest_path = out_dir / "latest.pt"
    if config["train"].get("resume", False) and latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        target_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"]) + 1
        best_valid_loss = float(checkpoint["best_valid_loss"])
        write_jsonl(log_path, {"event": "resume", "checkpoint": str(latest_path), "step": step})
    elif config.get("init_checkpoint"):
        checkpoint = torch.load(config["init_checkpoint"], map_location=device)
        target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        target_model.load_state_dict(checkpoint["model"])
        write_jsonl(
            log_path,
            {"event": "init_from_checkpoint", "checkpoint": str(config["init_checkpoint"])},
        )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    model.train()
    start_time = time.time()
    max_steps = config["train"]["max_steps"]
    grad_accum = config["train"]["gradient_accumulation_steps"]
    batch_size = config["data"]["batch_size"]

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
        for _ in range(grad_accum):
            x, y = train_data.get_batch(batch_size)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                _, loss = model(x, y)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.item()

        if config["optim"]["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["optim"]["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if step % config["train"]["log_interval"] == 0:
            elapsed = time.time() - start_time
            row = {
                "step": step,
                "loss": total_loss,
                "lr": lr,
                "elapsed_sec": elapsed,
                "tokens": (step + 1) * batch_size * model_config.context_length * grad_accum,
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
            best_valid_loss = min(best_valid_loss, losses["valid"])
            write_jsonl(
                log_path,
                {
                    "step": step,
                    "eval_train_loss": losses["train"],
                    "eval_valid_loss": losses["valid"],
                    "best_valid_loss": best_valid_loss,
                },
            )

        if step % config["train"]["save_interval"] == 0 or step == max_steps - 1:
            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(latest_path, target_model, optimizer, config, step, best_valid_loss)
            save_checkpoint(out_dir / f"step_{step:06d}.pt", target_model, optimizer, config, step, best_valid_loss)

        step += 1


if __name__ == "__main__":
    main()
