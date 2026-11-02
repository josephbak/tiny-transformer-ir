# %%
import torch
import torch.nn as nn
from torch.nn import functional as F

# %%
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)    # Q1: what goes in, what comes out?
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))   # Q2: what size?

    def forward(self, x):            # x: [B, T, n_embd]
        B, T, C = x.shape
        k = self.key(x)              # Q3: shape? [B, T, head_size]
        q = self.query(x)            # shape? [B, T, head_size]
        head_size = k.shape[-1]

        wei = q @ k.transpose(-2, -1) * head_size ** -0.5                    # Q4: scores from q and k, with scaling. shape: [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))                    # Q5: apply the mask. Which part of self.tril? what do you mean?
        wei = F.softmax(wei, dim=-1) 
        print(wei.sum(dim=-1))
        v = self.value(x)            # shape? [B, T, head_size]
        return wei @ v                   # Q6: output. shape [B, T, head_size]

"""
Name	Meaning	                                                            His final model
n_embd	width of each token’s vector: the $C$ in (B, T, C)	                64
head_size	width of one head’s query, key, and value vectors	            n_embd // n_head = 16
block_size	maximum context length: the most tokens the model sees at once	32
"""

# %%
x = torch.randn(4, 8, 32)
# print(Head(16)(x).shape)   # check shapes as you go
print (Head(32, 16, 8)(x).shape)

# %%
