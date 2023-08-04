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

  np.testing.assert_allclose(y_mps, y, atol=0.001)
  return mtl_t, mps_t

if __name__ == '__main__':
  # TODO: print FLOPS
  print("scaling N")
  for i in range(10):
    mtl_t, mps_t = sample(2**i, 128, 4, 16)
    print(f"N={2**i:3d}, HW=128, C=4, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling HW")
  for i in range(10):
    mtl_t, mps_t = sample(32, 32*(i+1), 4, 16)
    print(f"N=32, HW={32*i:3d}, C=4, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling C")
  for i in range(10):
    try: mtl_t, mps_t = sample(32, 128, 2**i, 16) # ???
    except: continue
    print(f"N=32, HW=128, C={2**i:3d}, F=16:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")

  print("\nscaling F")
  for i in range(10):
    mtl_t, mps_t = sample(32, 128, 4, 2**i)
    print(f"N=32, HW=128, C=4, F={2**i:3d}:\t\tmtl {mtl_t:.3f}s, mps: {mps_t:.3f}s, {mps_t/mtl_t:.3f}x")
