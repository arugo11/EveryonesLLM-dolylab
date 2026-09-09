import torch


class DataLoader:
    def __init__(self, text, config):
        self.config = config

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.ctoi = {char: index for index, char in enumerate(chars)}
        self.itoc = {index: char for index, char in enumerate(chars)}

        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.train_data, self.val_data = self.split_data()

    def encode(self, string):
        return [self.ctoi[char] for char in string]

    def decode(self, indices):
        return ''.join([self.itoc[index] for index in indices])

    def split_data(self):
        split_index = int(0.9 * len(self.data))
        return self.data[:split_index], self.data[split_index:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        start_indices = torch.randint(
            len(data) - self.config.input_sequence_length,
            (self.config.batch_size,),
        )

        input_sequences = torch.stack([
            data[start_index:start_index + self.config.input_sequence_length]
            for start_index in start_indices
        ])
        target_sequences = torch.stack([
            data[start_index + 1:start_index + self.config.input_sequence_length + 1]
            for start_index in start_indices
        ])

        input_sequences = input_sequences.to(self.config.device_type)
        target_sequences = target_sequences.to(self.config.device_type)
        return input_sequences, target_sequences
