import torch


class Config:
    batch_size = 16
    input_sequence_length = 512
    total_steps = 10_000
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluation_frequency = 100
    learning_rate = 0.001
    evaluation_loops = 10
    embedding_dim = 64
    hidden_dim = 256
    num_attention_heads = 4
    layer_count = 4
    dropout_rate = 0.1
    random_seed_value = 1337
