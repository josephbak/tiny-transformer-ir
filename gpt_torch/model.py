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
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))   # Q2: what size? [T, T]

    def forward(self, x):            # x: [B, T, n_embd]
        B, T, C = x.shape
        k = self.key(x)              # Q3: shape? [B, T, head_size]
        q = self.query(x)            # shape? [B, T, head_size]
        head_size = k.shape[-1]

        wei = q @ k.transpose(-2, -1) * head_size ** -0.5                    # Q4: scores from q and k, with scaling. shape: [B, T, T]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))                    # Q5: apply the mask. Which part of self.tril? what do you mean?
        wei = F.softmax(wei, dim=-1) 
        # print(wei.sum(dim=-1))
        v = self.value(x)            # shape? [B, T, head_size]
        return wei @ v                   # Q6: output. shape [B, T, head_size]

"""
Name	Meaning	                                                            His final model
n_embd	width of each token’s vector: the $C$ in (B, T, C)	                64
head_size	width of one head’s query, key, and value vectors	            n_embd // n_head = 16
block_size	maximum context length: the most tokens the model sees at once	32
"""

# %%
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head                                        # Q1: so that n_head heads add back up to n_embd
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])   # Q2: arguments?
        self.proj = nn.Linear(n_embd, n_embd)                        # Q3: output projection, in and out sizes?

    def forward(self, x):                                       # x: [B, T, n_embd]
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # Q4: run each head; concatenate along which dim?
                                                                #     shape after cat? [B, T, n_embd]
        return self.proj(out)                                   # shape? [B, T, n_embd]

# %%
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),     # Q1: expand. In and out sizes? (his model uses a 4× hidden width)
            nn.ReLU(),                # Q2: which activation? I chose relu
            nn.Linear(n_embd * 4, n_embd),     # Q3: project back. In and out sizes?
        )

    def forward(self, x):            # x: [B, T, n_embd]
        return self.net(x)                   # Q4: shape of the result? # [B, T, n_embd]
# %%
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa   = MultiHeadAttention(n_embd, n_head, block_size)   # Q1: arguments?
        self.ffwd = FeedForward(n_embd)                    # Q2: argument?
        self.ln1  = nn.LayerNorm(n_embd)                   # Q3: normalizes vectors of what size?
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x):                               # x: [B, T, n_embd]
        x = x + self.sa(self.ln1(x))                    # Q4: attention sublayer, pre-norm, with residual
        x = x + self.ffwd(self.ln2(x))                  # Q5: FFN sublayer, pre-norm, with residual
        return x                                        # shape?
# %%
class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding    = nn.Embedding(vocab_size, n_embd)    # Q1: one row per what? vocab how long is each row? embedding space
        self.position_embedding = nn.Embedding(block_size, n_embd)    # Q2: one row per what? block_size
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)                  # Q3: from what size to what size?

    def forward(self, idx, targets=None):                   # idx: [B, T] integer token ids
        B, T = idx.shape
        tok = self.token_embedding(idx)                                           # Q4: shape? [B, T, n_embd]
        pos = self.position_embedding(torch.arange(T))     # Q5: shape? (positions 0 … T−1)
        x = tok + pos                                             # Q6: combine tok and pos. Shape? [B, T, n_embd]
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                        # Q7: shape? [B, T, vocab_size]

        if targets is None:
            return logits, None
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T)) # Q8: cross-entropy; what reshaping does it need?
        return logits, loss

    def generate(self, idx, max_new_tokens):         # idx: [B, T] the starting context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]            # Q1: the model's position table has block_size rows.
                                                            # What must happen once idx grows longer than that?
            logits, _ = self(idx_cond)                      # shape?
            logits = logits[:, -1, :]                       # Q2: which position's logits predict the next token? shape?
            probs = F.softmax(logits, dim=-1)        # Q3: softmax over which axis?
            idx_next = torch.multinomial(probs, num_samples=1)   # shape? [B, 1]
            idx = torch.cat((idx, idx_next), dim=1) # Q4: append along which axis? [B, T+1]
        return idx

# %%
if __name__ == "__main__":
    x = torch.randn(4, 8, 32)
    print(Head(32, 16, 8)(x).shape)                   # [4, 8, 16]
    print(MultiHeadAttention(32, 4, 8)(x).shape)      # [4, 8, 32]
    print(FeedForward(32)(x).shape)                   # [4, 8, 32]
    print(Block(32, 4, 8)(x).shape)                   # [4, 8, 32]

    model = GPT(vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=8)
    idx = torch.randint(0, 65, (4, 8))
    logits, loss = model(idx, idx)
    print(logits.shape, loss)                         # [4, 8, 65], ≈ 4.2