import os
from torch.utils.data import Dataset
import torch

class LMDataset(Dataset):
    """
    Chunks a flat token stream into (input, target) pairs for
    next-token prediction.

    input:  tokens[i   : i + block_size]
    target: tokens[i+1 : i + block_size + 1]
    """

    def __init__(self, token_ids, block_size):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return (len(self.token_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.token_ids[start     : start + self.block_size]
        y = self.token_ids[start + 1 : start + self.block_size + 1]
        return x, y


class HFCausalLMDataset(Dataset):
    """Wrap fixed-width token blocks from a Hugging Face Dataset."""

    def __init__(self, hf_dataset, block_size, column="input_ids"):
        self.hf_dataset = hf_dataset
        self.block_size = block_size
        self.column = column

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        token_ids = self.hf_dataset[idx][self.column]
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            token_ids = token_ids.to(dtype=torch.long)

        expected_width = self.block_size + 1
        if token_ids.numel() != expected_width:
            raise ValueError(
                f"Expected token block width {expected_width}, got {token_ids.numel()}"
            )

        return token_ids[:-1], token_ids[1:]
