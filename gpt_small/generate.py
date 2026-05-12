from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt_small.tokenizer import Tokenizer
from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.training.utils import resolve_device


def parse_prompt(prompt: str) -> list[int]:
    return [int(part) for part in prompt.strip().split()] if prompt.strip() else [0]


class TextTokenizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.fast_tokenizer = None
        self.simple_tokenizer = None
        self.eot_token = "<|endoftext|>"

        if args.tokenizer_json_path.exists():
            from tokenizers import Tokenizer as FastTokenizer

            self.fast_tokenizer = FastTokenizer.from_file(str(args.tokenizer_json_path))
            self.eot_id = self.fast_tokenizer.token_to_id(self.eot_token)
        elif args.vocab_path.exists() and args.merges_path.exists():
            self.simple_tokenizer = Tokenizer.from_files(
                args.vocab_path,
                args.merges_path,
                special_tokens=[self.eot_token],
            )
            self.eot_id = self.simple_tokenizer.token_to_id.get(self.eot_token.encode("utf-8"))
        else:
            self.eot_id = None

    @property
    def available(self) -> bool:
        return self.fast_tokenizer is not None or self.simple_tokenizer is not None

    def encode(self, text: str) -> list[int]:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.encode(text, add_special_tokens=False).ids
        if self.simple_tokenizer is not None:
            return self.simple_tokenizer.encode(text)
        raise RuntimeError("No tokenizer artifacts found for text prompt encoding.")

    def decode(self, ids: list[int]) -> str:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.decode(ids, skip_special_tokens=False)
        if self.simple_tokenizer is not None:
            return self.simple_tokenizer.decode(ids)
        return " ".join(str(token_id) for token_id in ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--ids", action="store_true", help="Interpret --prompt as space-separated token ids.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer-json-path", type=Path, default=Path("artifacts/tokenizer/tokenizer.json"))
    parser.add_argument("--vocab-path", type=Path, default=Path("artifacts/tokenizer/vocab.bin"))
    parser.add_argument("--merges-path", type=Path, default=Path("artifacts/tokenizer/merges.bin"))
    parser.add_argument("--stop-at-eot", action="store_true")
    parser.add_argument("--print-ids", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = TransformerLM(GPTConfig(**checkpoint["config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = TextTokenizer(args)
    if args.ids:
        input_ids = parse_prompt(args.prompt)
    elif tokenizer.available:
        input_ids = tokenizer.encode(args.prompt)
        if not input_ids:
            input_ids = [tokenizer.eot_id if tokenizer.eot_id is not None else 0]
    else:
        input_ids = parse_prompt(args.prompt)

    prompt_len = len(input_ids)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    output_ids = out[0].tolist()
    if args.stop_at_eot and tokenizer.eot_id is not None:
        for index in range(prompt_len, len(output_ids)):
            if output_ids[index] == tokenizer.eot_id:
                output_ids = output_ids[: index + 1]
                break

    if args.print_ids or not tokenizer.available:
        print(" ".join(str(token) for token in output_ids))
    else:
        print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
