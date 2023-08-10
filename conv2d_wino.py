from typing import Tuple
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram
from tinygrad.helpers import dtypes
from string import Template
import functools

BHW = 8
RLOOP = 8

_src_tmpl = Template(open("conv2d_wino_2x2_3x3.metal", "r").read()) 

@functools.lru_cache(maxsize=None)
def _prgs(N, HW, C, F):
  src = _src_tmpl.substitute(N=N,HW=HW,C=C,F=F,BHW=BHW,RLOOP=RLOOP)
  return MetalProgram("filter_transform", src), MetalProgram("conv2", src)

def conv2d_wino(ims:RawMetalBuffer, fs:RawMetalBuffer, size:Tuple[int,int,int,int]) -> RawMetalBuffer:
  N, HW, C, F = size
  tfs = RawMetalBuffer(F*C*4*4, dtypes.float32)
  out = RawMetalBuffer(N*F*(HW-2)*(HW-2), dtypes.float32)
  ft_prg, conv_prg = _prgs(N, HW, C, F)
  ft_prg([F,1,1], [C,1,1], tfs, fs, wait=True)
  # conv_prg([1, HW/BHW, N], [HW//BHW, 1, 1], out, ims, tfs, wait=True)
  conv_prg([HW//BHW, HW//BHW, N], [1, 1, C/RLOOP], out, ims, tfs, wait=True)
  return out
