from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gpt_small.tokenizer import Tokenizer


IGNORE_INDEX = -100
EOT_TOKEN = "<|endoftext|>"


class TextTokenizer:
    def __init__(
        self,
        tokenizer_json_path: str | Path | None = None,
        vocab_path: str | Path | None = None,
        merges_path: str | Path | None = None,
    ) -> None:
        self.fast_tokenizer = None
        self.simple_tokenizer = None
        self.eot_token = EOT_TOKEN

        json_path = Path(tokenizer_json_path) if tokenizer_json_path else None
        vocab = Path(vocab_path) if vocab_path else None
        merges = Path(merges_path) if merges_path else None

        if json_path is not None and json_path.exists():
            from tokenizers import Tokenizer as FastTokenizer

            self.fast_tokenizer = FastTokenizer.from_file(str(json_path))
            self.eot_id = self.fast_tokenizer.token_to_id(self.eot_token)
        elif vocab is not None and merges is not None and vocab.exists() and merges.exists():
            self.simple_tokenizer = Tokenizer.from_files(vocab, merges, special_tokens=[self.eot_token])
            self.eot_id = self.simple_tokenizer.token_to_id.get(self.eot_token.encode("utf-8"))
        else:
            raise FileNotFoundError("No tokenizer artifacts found for SFT text encoding.")

        if self.eot_id is None:
            raise ValueError(f"Tokenizer does not contain required special token {self.eot_token!r}")

    def encode(self, text: str) -> list[int]:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.encode(text, add_special_tokens=False).ids
        if self.simple_tokenizer is not None:
            return self.simple_tokenizer.encode(text)
        raise RuntimeError("Tokenizer is not initialized.")

    def decode(self, ids: list[int]) -> str:
        if self.fast_tokenizer is not None:
            return self.fast_tokenizer.decode(ids, skip_special_tokens=False)
        if self.simple_tokenizer is not None:
            return self.simple_tokenizer.decode(ids)
        return " ".join(str(token_id) for token_id in ids)


def _normalize_role(role: str) -> str:
    role = role.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "gpt", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"unsupported message role {role!r}")


def _format_prefix(role: str) -> str:
    if role == "system":
        return "System: "
    if role == "user":
        return "User: "
    if role == "assistant":
        return "Assistant: "
    raise ValueError(f"unsupported role {role!r}")


def _append_text(
    input_ids: list[int],
    labels: list[int],
    tokenizer: TextTokenizer,
    text: str,
    train_on_text: bool,
) -> None:
    ids = tokenizer.encode(text)
    input_ids.extend(ids)
    labels.extend(ids if train_on_text else [IGNORE_INDEX] * len(ids))


def encode_messages(
    messages: list[dict[str, Any]],
    tokenizer: TextTokenizer,
    train_eot: bool = True,
) -> tuple[list[int], list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []

    for index, message in enumerate(messages):
        role = _normalize_role(str(message["role"]))
        content = str(message.get("content", "")).strip()
        if not content:
            continue

        prefix = _format_prefix(role)
        suffix = "\n" if index < len(messages) - 1 else ""
        train_on_text = role == "assistant"
        _append_text(input_ids, labels, tokenizer, prefix, train_on_text=False)
        _append_text(input_ids, labels, tokenizer, content + suffix, train_on_text=train_on_text)

    input_ids.append(tokenizer.eot_id)
    labels.append(tokenizer.eot_id if train_eot else IGNORE_INDEX)
    return input_ids, labels


def encode_prompt_response(
    prompt: str,
    response: str,
    tokenizer: TextTokenizer,
    train_eot: bool = True,
) -> tuple[list[int], list[int]]:
    return encode_messages(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenizer,
        train_eot=train_eot,
    )


def encode_sft_row(
    row: dict[str, Any],
    tokenizer: TextTokenizer | None,
    train_eot: bool = True,
) -> tuple[list[int], list[int]]:
    if "input_ids" in row:
        input_ids = [int(token) for token in row["input_ids"]]
        if "labels" in row:
            labels = [int(token) for token in row["labels"]]
        elif "label_mask" in row:
            mask = [bool(value) for value in row["label_mask"]]
            labels = [token if keep else IGNORE_INDEX for token, keep in zip(input_ids, mask)]
        else:
            labels = input_ids.copy()
        if len(input_ids) != len(labels):
            raise ValueError("input_ids and labels must have the same length")
        return input_ids, labels

    if tokenizer is None:
        raise ValueError("text SFT rows require tokenizer artifacts")

    if "messages" in row:
        return encode_messages(row["messages"], tokenizer, train_eot=train_eot)

    if "prompt" in row and "response" in row:
        return encode_prompt_response(
            str(row["prompt"]),
            str(row["response"]),
            tokenizer,
            train_eot=train_eot,
        )

    if "instruction" in row and "output" in row:
        prompt = str(row["instruction"])
        if row.get("input"):
            prompt = f"{prompt}\n{row['input']}"
        return encode_prompt_response(prompt, str(row["output"]), tokenizer, train_eot=train_eot)

    raise ValueError("SFT row must contain messages, prompt/response, instruction/output, or input_ids")


class SFTJsonlDataset:
    def __init__(
        self,
        path: str | Path,
        context_length: int,
        tokenizer: TextTokenizer | None = None,
        pad_token_id: int | None = None,
        train_eot: bool = True,
        truncate: bool = True,
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else (
            tokenizer.eot_id if tokenizer is not None else 0
        )
        self.device = torch.device(device)
        self.examples: list[tuple[list[int], list[int]]] = []

        max_tokens = context_length + 1
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                input_ids, labels = encode_sft_row(row, tokenizer, train_eot=train_eot)
                if len(input_ids) > max_tokens:
                    if not truncate:
                        raise ValueError(
                            f"{self.path}:{line_no} has {len(input_ids)} tokens, "
                            f"exceeding context_length + 1 ({max_tokens})"
                        )
                    input_ids = input_ids[:max_tokens]
                    labels = labels[:max_tokens]
                if len(input_ids) < 2:
                    continue
                if all(label == IGNORE_INDEX for label in labels[1:]):
                    continue
                self.examples.append((input_ids, labels))

        if not self.examples:
            raise ValueError(f"{self.path} did not contain any trainable SFT examples")

    def __len__(self) -> int:
        return len(self.examples)

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self.examples), (batch_size,))
        rows = [self.examples[index] for index in indices.tolist()]
        max_len = min(max(len(input_ids) for input_ids, _ in rows), self.context_length + 1)

        x_batch = []
        y_batch = []
        for input_ids, labels in rows:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_len
                labels = labels + [IGNORE_INDEX] * pad_len
            x_batch.append(input_ids[:-1])
            y_batch.append(labels[1:])

        return (
            torch.tensor(x_batch, dtype=torch.long, device=self.device),
            torch.tensor(y_batch, dtype=torch.long, device=self.device),
        )
