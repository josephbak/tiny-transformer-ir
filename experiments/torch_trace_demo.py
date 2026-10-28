import torch


def f(x):
    if x.sum() > 0:
        return x * 2
    return x * -1


traced = torch.jit.trace(f, torch.tensor([1., 2.]))   # the example input takes the "> 0" branch

print(traced(torch.tensor([1., 2.])))     # tensor([2., 4.])    correct
print(traced(torch.tensor([-1., -2.])))   # tensor([-2., -4.])  WRONG: should be [1., 2.]