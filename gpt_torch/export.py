# %%
from pathlib import Path

import torch
from model import GPT

model = GPT(vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=8).eval()
print(type(model.blocks[0].sa).__name__)       # expect FusedMultiHeadAttention

idx = torch.randint(0, 65, (4, 6))             # T = 6, so T, head_size, and block_size are distinguishable
# %%
exported = torch.export.export(model, (idx,))


# save to <repo>/ir/torch/, whatever directory this runs from
out_dir = Path(__file__).resolve().parent.parent / "ir" / "torch"
out_dir.mkdir(parents=True, exist_ok=True)

def clean(text):
    return text.replace(str(Path.home()), "~")    # /Users/<you>/... → ~/...

# the same program, decomposed into Core ATen (closer to the level of a jaxpr)
decomposed = exported.run_decompositions()

(out_dir / "gpt_export.txt").write_text(clean(exported.graph_module.print_readable(print_output=False)))
(out_dir / "gpt_export_core_aten.txt").write_text(clean(decomposed.graph_module.print_readable(print_output=False)))

print("wrote", out_dir / "gpt_export.txt", "and", out_dir / "gpt_export_core_aten.txt")
# %%
