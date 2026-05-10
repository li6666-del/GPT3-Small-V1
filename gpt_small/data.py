from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
    "int64": np.int64,
}


class MemmapTokenDataset:
    def __init__(
        self,
        path: str | Path,
        context_length: int,
        dtype: str = "uint16",
        device: str | torch.device = "cpu",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype}; choose one of {sorted(DTYPES)}")
        self.tokens = np.memmap(self.path, dtype=DTYPES[dtype], mode="r")
        self.context_length = context_length
        self.device = torch.device(device)
        if len(self.tokens) <= context_length:
            raise ValueError(f"{self.path} needs more than {context_length} tokens")

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(0, len(self), (batch_size,))
        x = np.stack([self.tokens[i : i + self.context_length] for i in starts.tolist()])
        y = np.stack(
            [self.tokens[i + 1 : i + self.context_length + 1] for i in starts.tolist()]
        )
        return (
            torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True),
            torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True),
        )
