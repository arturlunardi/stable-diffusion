import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: [b_size, seq_len, dim]
        input_shape = x.shape
        b_size, sequence_length, d_embed = input_shape.shape

        intermim_shape = (b_size, sequence_length, self.n_heads, self.d_head)

        # [b_size, seq_len, dim] -> [b_size, seq_len, dim * 3] -> 3 tensors [b_size, seq_len, dim]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # [b_size, seq_len, dim] -> [b_size, seq_len, h, dim / h] -> [b_size, h, seq_len, dim / h]
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)

        # [b_size, h, seq_len, seq_len]
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # [b_size, h, seq_len, seq_len] @ [b_size, h, seq_len, dim / h] -> [b_size, h, seq_len, dim / h]
        output = weight @ v

        # [b_size, h, seq_len, dim / h] -> [b_size, seq_len, h, dim / h]
        output = output.transpose(1, 2)
        # [b_size, seq_len, dim]
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # [b_size, seq_len, d_embed]
        return output
