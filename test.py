from conv2d_wino import conv2d_wino
import numpy as np
import torch
import torch.nn.functional as torchf
from tinygrad.runtime.ops_metal import RawMetalBuffer

cases = [(1,128,16,1), (1,32,1,1)]

for N, HW, C, F in cases:
  nims = np.random.default_rng().standard_normal(size=(N,C,HW,HW), dtype=np.float32)
  nfs = np.random.default_rng().standard_normal(size=(F,C,3,3), dtype=np.float32)

  tims = torch.from_numpy(nims).to('mps')
  tfs = torch.from_numpy(nfs).to('mps')
  y_torch = torchf.conv2d(tims, tfs).cpu().numpy()

  ims = RawMetalBuffer.fromCPU(nims)
  fs = RawMetalBuffer.fromCPU(nfs)
  y = conv2d_wino(ims, fs, (N,HW,C,F)).toCPU().reshape(N,F,HW-2,HW-2)

  np.testing.assert_allclose(y_torch, y, atol=0.001)
