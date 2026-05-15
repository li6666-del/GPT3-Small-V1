from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpt_small.model import GPTConfig, TransformerLM
from gpt_small.sft_data import TextTokenizer
from gpt_small.training.utils import resolve_device, safe_torch_load


DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "sft-v4189-00-usable_core_checkpoint" / "step_000055.pt"
DEFAULT_TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
EXIT_COMMANDS = {"/q", "/quit", "/exit", "q", "quit", "exit", "退出"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive chat with a GPT-small SFT checkpoint.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-json-path", type=Path, default=DEFAULT_TOKENIZER_DIR / "tokenizer.json")
    parser.add_argument("--vocab-path", type=Path, default=DEFAULT_TOKENIZER_DIR / "vocab.bin")
    parser.add_argument("--merges-path", type=Path, default=DEFAULT_TOKENIZER_DIR / "merges.bin")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps, ...")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument(
        "--history-turns",
        type=int,
        default=0,
        help="How many previous turns to include. Default 0 matches the SFT single-turn format.",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--once", default="", help="Run one prompt and exit.")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_tokenizer(args: argparse.Namespace) -> TextTokenizer:
    tokenizer_json_path = args.tokenizer_json_path.expanduser().resolve()
    vocab_path = args.vocab_path.expanduser().resolve()
    merges_path = args.merges_path.expanduser().resolve()
    fast_tokenizer_available = importlib.util.find_spec("tokenizers") is not None
    if tokenizer_json_path.exists() and not fast_tokenizer_available and vocab_path.exists() and merges_path.exists():
        tokenizer_json_path = Path("__missing_tokenizer_json__")
    try:
        return TextTokenizer(
            tokenizer_json_path=tokenizer_json_path,
            vocab_path=vocab_path,
            merges_path=merges_path,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Tokenizer artifacts are missing. Expected either:\n"
            f"  {tokenizer_json_path}\n"
            "or both:\n"
            f"  {vocab_path}\n"
            f"  {merges_path}\n"
            "Copy the original tokenizer into artifacts/tokenizer, or pass the three tokenizer paths explicitly."
        ) from exc


def load_model(checkpoint_path: Path, device: torch.device) -> TransformerLM:
    checkpoint_path = require_file(checkpoint_path, "checkpoint")
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if "config" not in checkpoint or "model" not in checkpoint["config"]:
        raise ValueError(f"checkpoint does not contain config.model: {checkpoint_path}")
    model = TransformerLM(GPTConfig(**checkpoint["config"]["model"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def build_prompt(
    user_text: str,
    history: list[tuple[str, str]],
    history_turns: int,
    system_prompt: str,
) -> str:
    parts: list[str] = []
    if system_prompt.strip():
        parts.append(f"System: {system_prompt.strip()}")
    if history_turns > 0:
        for user, assistant in history[-history_turns:]:
            parts.append(f"User: {user.strip()}")
            parts.append(f"Assistant: {assistant.strip()}")
    parts.append(f"User: {user_text.strip()}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def clean_reply(text: str, eot_token: str) -> str:
    text = text.replace(eot_token, "")
    stop_markers = ["\nUser:", "\nAssistant:", "\nSystem:", "User:", "Assistant:", "System:"]
    first_stop = len(text)
    for marker in stop_markers:
        index = text.find(marker)
        if index >= 0:
            first_stop = min(first_stop, index)
    return text[:first_stop].strip()


@torch.no_grad()
def generate_reply(
    model: TransformerLM,
    tokenizer: TextTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
) -> str:
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [tokenizer.eot_id]

    max_prompt_tokens = max(1, model.config.context_length - max_new_tokens)
    if len(input_ids) > max_prompt_tokens:
        input_ids = input_ids[-max_prompt_tokens:]

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    new_ids = out[0].tolist()[len(input_ids) :]
    if tokenizer.eot_id in new_ids:
        new_ids = new_ids[: new_ids.index(tokenizer.eot_id)]
    return clean_reply(tokenizer.decode(new_ids), tokenizer.eot_token)


def run_once(args: argparse.Namespace, model: TransformerLM, tokenizer: TextTokenizer, device: torch.device) -> None:
    prompt = build_prompt(args.once, [], args.history_turns, args.system_prompt)
    reply = generate_reply(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(reply)


def run_chat(args: argparse.Namespace, model: TransformerLM, tokenizer: TextTokenizer, device: torch.device) -> None:
    history: list[tuple[str, str]] = []
    print("已加载 checkpoint。输入 /exit 退出，/reset 清空历史。")
    print("默认是单轮问答；需要多轮上下文可重启时加 --history-turns N。")
    while True:
        try:
            user_text = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_text:
            continue
        if user_text in EXIT_COMMANDS:
            return
        if user_text == "/reset":
            history.clear()
            print("历史已清空。")
            continue

        prompt = build_prompt(user_text, history, args.history_turns, args.system_prompt)
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"模型> {reply}")
        history.append((user_text, reply))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(args)
    model = load_model(args.checkpoint, device)
    print(f"device={device}, checkpoint={args.checkpoint}")

    if args.once:
        run_once(args, model, tokenizer, device)
    else:
        run_chat(args, model, tokenizer, device)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
