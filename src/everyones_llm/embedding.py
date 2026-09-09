import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_indices):
        return self.token_embedding_table(input_indices)


class PositionEmbedding(nn.Module):
    def __init__(self, input_sequence_length=8, embedding_dim=8):
        super().__init__()
        self.position_embedding_table = nn.Embedding(input_sequence_length, embedding_dim)

    def forward(self, input_indices):
        sequence_length = input_indices.shape[1]
        position_indices = torch.arange(sequence_length, device=input_indices.device)
        return self.position_embedding_table(position_indices)


class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.token_embedding_layer = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.position_embedding_layer = PositionEmbedding(
            input_sequence_length=config.input_sequence_length,
            embedding_dim=config.embedding_dim,
        )

    def forward(self, input_indices):
        token_embedding = self.token_embedding_layer(input_indices)
        position_embedding = self.position_embedding_layer(input_indices)
        return token_embedding + position_embedding
