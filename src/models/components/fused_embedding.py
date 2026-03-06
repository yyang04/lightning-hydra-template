import numpy as np
import torch
from torch import nn


class FusedEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.offsets = torch.tensor([0, *np.cumsum(field_dims)[:-1]], dtype=torch.long)
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.register_buffer('offset_buffer', self.offsets)

    def forward(self, x):
        x = x.long()
        x = x + self.offset_buffer
        return self.embedding(x)
