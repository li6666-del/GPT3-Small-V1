from __future__ import annotations

import argparse

import torch

from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.training.utils import resolve_device


def parse_prompt(prompt: str) -> list[int]:
    return [int(part) for part in prompt.strip().split()] if prompt.strip() else [0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TransformerLM(GPTConfig(**checkpoint["config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = torch.tensor([parse_prompt(args.prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(" ".join(str(token) for token in out[0].tolist()))


if __name__ == "__main__":
    main()
