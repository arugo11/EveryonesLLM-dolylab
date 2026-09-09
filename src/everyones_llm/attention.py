import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionHead(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key_fc = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.query_fc = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.value_fc = nn.Linear(config.embedding_dim, head_size, bias=False)
        self.head_size = head_size
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_tensor):
        B, T, C = input_tensor.shape

        Key = self.key_fc.forward(input_tensor)
        Query = self.query_fc.forward(input_tensor)
        Value = self.value_fc.forward(input_tensor)

        attention_weights_before_mask = Query @ Key.transpose(-2, -1) * self.head_size**(-0.5)

        mask = torch.triu(torch.ones(T, T), diagonal=1).to(input_tensor.device)
        masked_attention_weights = attention_weights_before_mask.masked_fill(mask == 1, float('-inf'))

        attention_weights = F.softmax(masked_attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = attention_weights @ Value
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.embedding_dim = config.embedding_dim
        self.head_size = int(self.embedding_dim / self.num_attention_heads)

        self.attention_heads = nn.ModuleList([
            AttentionHead(self.head_size, config)
            for _ in range(self.num_attention_heads)
        ])

        self.output_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, input_tensor):
        head_outputs_list = [head.forward(input_tensor) for head in self.attention_heads]
        concatenated = torch.cat(head_outputs_list, dim=-1)
        projected = self.output_projection.forward(concatenated)
        return self.dropout(projected)
