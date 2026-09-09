import torch


class WorkshopConfig:
    def __init__(self):
        self.batch_size = 16
        self.input_sequence_length = 512
        self.total_steps = 2000
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.evaluation_frequency = 100
        self.learning_rate = 0.001
        self.evaluation_loops = 10
        self.embedding_dim = 512
        self.hidden_dim = 2048
        self.num_attention_heads = 8
        self.layer_count = 4
        self.dropout_rate = 0.1
        self.random_seed_value = 1337
        self.use_bf16 = False

    def as_dict(self):
        return dict(vars(self))

    @classmethod
    def from_dict(cls, values):
        config = cls()
        for key, value in values.items():
            setattr(config, key, value)
        return config
