import torch
import torch.nn.functional as NNF
import time

assert torch.backends.mps.is_available()

def sample(N, HW, C, F):
  ims = torch.rand(N, C, HW, HW, device="mps")
  fs = torch.randn(F, C, 3, 3, device="mps")

  st = time.monotonic()
  y = NNF.conv2d(ims, fs)
  print(y.shape)
  return y, time.monotonic()-st

if __name__ == '__main__':
  print(sample(32, 4096, 4, 2)[1]) # y.shape: [2, 2, 4094, 4094]
