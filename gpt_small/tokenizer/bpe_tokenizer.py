from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        if special_tokens is None:
            special_tokens = []

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.token_to_id = {token: idx for idx, token in vocab.items()}
        self.merges_rank = {pair: rank for rank, pair in enumerate(merges)}

        if special_tokens:
            special_tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in special_tokens_sorted]
            self.special_pattern = "(" + "|".join(escaped) + ")"
        else:
            self.special_pattern = None

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        merges_path: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with Path(vocab_path).open("rb") as f:
            loaded_vocab = pickle.load(f)
        with Path(merges_path).open("rb") as f:
            merges = pickle.load(f)

        if isinstance(loaded_vocab, dict):
            vocab = loaded_vocab
        else:
            vocab = {idx: token for idx, token in enumerate(loaded_vocab)}

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def decode(self, ids: list[int]) -> str:
        token_bytes = []
        for idx in ids:
            token_bytes.append(self.vocab[idx])
        return b"".join(token_bytes).decode("utf-8", errors="replace")

    @lru_cache(maxsize=None)
    def _bpe_encode_bytes(self, token_bytes: bytes) -> tuple[bytes, ...]:
        parts = tuple(bytes([byte]) for byte in token_bytes)
        if len(parts) <= 1:
            return parts

        while True:
            best_pair = None
            best_rank = float("inf")

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.merges_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = tuple(new_parts)

            if len(parts) <= 1:
                break

        return parts

    def _encode_ordinary_text(self, text: str) -> list[int]:
        ids = []
        for match in re.finditer(PAT, text):
            token_bytes = match.group(0).encode("utf-8")
            bpe_tokens = self._bpe_encode_bytes(token_bytes)
            for token in bpe_tokens:
                if token not in self.token_to_id:
                    raise KeyError(f"Token {token!r} not found in vocab")
                ids.append(self.token_to_id[token])
        return ids

    def encode(self, text: str) -> list[int]:
        if self.special_pattern is None:
            return self._encode_ordinary_text(text)

        ids = []
        chunks = re.split(self.special_pattern, text)
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                special_bytes = chunk.encode("utf-8")
                if special_bytes not in self.token_to_id:
                    raise KeyError(
                        f"Special token {chunk!r} is not in vocab. "
                        "Make sure it was added during BPE training."
                    )
                ids.append(self.token_to_id[special_bytes])
            else:
                ids.extend(self._encode_ordinary_text(chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
