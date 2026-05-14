from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.sft_data import TextTokenizer
from gpt_small.training.sft import load_generation_prompts, load_model_weights, run_generation_eval
from gpt_small.training.utils import load_json, resolve_device, resolve_dtype, set_seed, write_jsonl


def build_tokenizer(config: dict[str, Any]) -> TextTokenizer:
    data_cfg = config["data"]
    return TextTokenizer(
        tokenizer_json_path=data_cfg.get("tokenizer_json_path"),
        vocab_path=data_cfg.get("vocab_path"),
        merges_path=data_cfg.get("merges_path"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_json(args.config)
    set_seed(int(config.get("seed", 20260514)))

    device = resolve_device(config.get("device", "auto"))
    _amp_dtype = resolve_dtype(config.get("dtype", "bfloat16"), device)
    model = TransformerLM(GPTConfig(**config["model"])).to(device)
    load_model_weights(model, config["checkpoint"], device)
    tokenizer = build_tokenizer(config)

    generation_cfg = dict(config["generation_eval"])
    prompts = load_generation_prompts(generation_cfg["prompts_path"])
    output_path = Path(generation_cfg["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.get("fresh", True) and output_path.exists():
        output_path.unlink()

    step = int(generation_cfg.get("step", 0))
    run_generation_eval(model, tokenizer, prompts, generation_cfg, output_path, step, device)

    log_path = Path(config.get("log_path", output_path.parent / "sft_log.jsonl"))
    write_jsonl(
        log_path,
        {
            "event": "checkpoint_generation_eval",
            "checkpoint": config["checkpoint"],
            "prompts": len(prompts),
            "output_path": str(output_path),
            "step": step,
            "device": str(device),
        },
    )
    print(json.dumps({"status": "done", "prompts": len(prompts), "output_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
