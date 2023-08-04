import time
import numpy as np
import torch
import torch.mps
import torch.nn.functional as torchf
from conv2d_wino import conv2d_wino
from tinygrad.runtime.ops_metal import RawMetalBuffer

# TODO: sync devices. make sure that i am accurately getting what i need

def sample(N, HW, C, F):
  nims = np.random.default_rng().standard_normal(size=(N,C,HW,HW), dtype=np.float32)
  nfs = np.random.default_rng().standard_normal(size=(F,C,3,3), dtype=np.float32)

  ims = RawMetalBuffer.fromCPU(nims)
  fs = RawMetalBuffer.fromCPU(nfs)
  st = time.perf_counter()
  y = conv2d_wino(ims, fs, (N,HW,C,F)).toCPU().reshape(N,F,HW-2,HW-2)
  mtl_t = time.perf_counter()-st

  tims = torch.from_numpy(nims).to('mps')
  tfs = torch.from_numpy(nfs).to('mps')
  st = time.perf_counter()
  y_mps = torchf.conv2d(tims, tfs).cpu().numpy()
  mps_t = time.perf_counter()-st

  # np.set_printoptions(suppress=True,precision=3,linewidth=300)
  # print(y)
  # print(y_mps)

  np.testing.assert_allclose(y_mps, y, atol=0.001)
  return mtl_t, mps_t

# cases = [(1,128,16,1), (1,32,1,1), (1,32,1,2), (16,16,16,16), (32,128,4,512)]
cases = [(1,32,1,1), (1,32,1,1), (1,32,1,2), (16,16,16,16), (32,128,4,512)]

for args in cases: sample(*args)
