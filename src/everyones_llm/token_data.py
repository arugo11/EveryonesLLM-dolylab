from pathlib import Path

import numpy as np
import torch


class TokenDataLoader:
    def __init__(self, data_dir, config):
        self.config = config
        data_dir = Path(data_dir)
        self.train_data = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
        self.val_data = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        max_start = len(data) - self.config.input_sequence_length - 1
        if max_start <= 0:
            raise ValueError(f'{split}.bin is too small for sequence length {self.config.input_sequence_length}.')
        start_indices = torch.randint(max_start, (self.config.batch_size,))
        input_sequences = torch.stack([
            torch.from_numpy(np.asarray(data[i:i + self.config.input_sequence_length], dtype=np.int64))
            for i in start_indices.tolist()
        ])
        target_sequences = torch.stack([
            torch.from_numpy(np.asarray(data[i + 1:i + self.config.input_sequence_length + 1], dtype=np.int64))
            for i in start_indices.tolist()
        ])
        return (
            input_sequences.to(self.config.device_type),
            target_sequences.to(self.config.device_type),
        )
