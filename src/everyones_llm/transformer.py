import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)
        self.multihead_attention = MultiHeadAttention(config=config)
        self.feed_forward = FeedForward(config=config)

    def forward(self, input_tensor):
        normed_input = self.layer_norm1(input_tensor)
        attention_output = self.multihead_attention(normed_input)
        residual_attention = attention_output + input_tensor
        normed_attention = self.layer_norm2(residual_attention)
        feedforward_output = self.feed_forward(normed_attention)
        final_output = feedforward_output + residual_attention
        return final_output
