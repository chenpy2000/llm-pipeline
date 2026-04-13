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
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx):
        x = self.token_ids[idx     : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y
    