import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import EmbeddingModule
from .transformer import TransformerBlock


class VocabularyLogits(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.vocab_projection = nn.Linear(config.embedding_dim, vocab_size)

    def forward(self, transformer_block_output):
        normalized_output = self.output_norm.forward(transformer_block_output)
        vocab_logits = self.vocab_projection.forward(normalized_output)
        return vocab_logits


class nanoGPT(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.embedding = EmbeddingModule(vocab_size, config=config)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config=config)
            for _ in range(config.layer_count)
        ])
        self.vocab_projection = VocabularyLogits(vocab_size=vocab_size, config=config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_indices, target_indices=None):
        embeddings = self.embedding(input_indices)
        blocks_output = self.blocks(embeddings)
        logits = self.vocab_projection(blocks_output)

        if target_indices is None:
            return logits, None

        batch_size, token_len, vocab_size = logits.shape
        logits_for_loss = logits.view(batch_size * token_len, vocab_size)
        targets = target_indices.view(batch_size * token_len)
        loss = self.criterion(logits_for_loss, targets)
        return logits, loss

    def generate(self, input_indices, max_new_tokens):
        for _ in range(max_new_tokens):
            input_conditioned = input_indices[:, -self.config.input_sequence_length:]
            logits, _ = self.forward(input_conditioned, target_indices=None)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_indices = torch.cat((input_indices, next_token), dim=1)
        return input_indices
