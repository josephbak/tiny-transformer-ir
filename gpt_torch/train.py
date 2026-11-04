# %%
# Train the GPT from model.py on Tiny Shakespeare (character-level), following Karpathy's lecture.
# Get the data first (macOS and Linux both have curl), from the repo root:
#   curl -o gpt_torch/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# Run:  uv run python gpt_torch/train.py     (or cell by cell in VS Code)

from pathlib import Path

import torch

from model import GPT   # works because Python adds this script's folder (gpt_torch/) to the import path

# %%
# hyperparameters: his final notebook's sizes, with a short run so it's light on a CPU
batch_size = 16      # sequences per batch (B)
block_size = 32      # context length (T at most)
n_embd = 64
n_head = 4
n_layer = 4
max_iters = 500      # his run used 5000; 500 is enough to watch the loss fall
eval_interval = 100
eval_iters = 50      # batches averaged per loss estimate (his used 200)
learning_rate = 1e-3
device = "cpu"       # the model's torch.arange(T) has no device= yet, so keep everything on the CPU
torch.manual_seed(1337)

# %%
# data: characters ↔ integer ids
text = (Path(__file__).parent / "input.txt").read_text(encoding="utf-8")
chars = sorted(set(text))
vocab_size = len(chars)                         # 65
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]         # string → list of ids
decode = lambda ids: "".join(itos[i] for i in ids)   # list of ids → string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]      # first 90% train, last 10% validation


def get_batch(split):
    """B random windows of block_size tokens (x), and the same windows shifted by one (y)."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()                                # measurement only: no gradient bookkeeping
def estimate_loss(model):
    """Average loss over eval_iters batches, for both splits."""
    out = {}
    model.eval()                                # evaluation mode (matters once dropout is used)
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()                               # back to training mode
    return out


# %%
# model and optimizer
model = GPT(vocab_size, n_embd, n_head, n_layer, block_size).to(device)
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M parameters")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
# training loop
for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)                     # forward
    optimizer.zero_grad(set_to_none=True)       # clear old gradients
    loss.backward()                             # backward: gradients for every weight
    optimizer.step()                            # update the weights

# %%
# generate from the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)   # start from id 0 (newline)
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))