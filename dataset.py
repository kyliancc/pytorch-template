import torch
from torch.utils.data import Dataset

import random


class TemplateDataset(Dataset):
    def __init__(self, set_type: str = 'train'):
        if set_type == 'train':
            self.data = [(torch.randn([3, 28, 28]), random.randint(0, 9)) for _ in range(500)]
        if set_type == 'val':
            self.data = [(torch.randn([3, 28, 28]), random.randint(0, 9)) for _ in range(100)]

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)
