from tinygrad.tensor import Tensor
from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram

def conv2d(ims:RawMetalBuffer, fs:RawMetalBuffer):
  src = open("conv2d_wino_2x2_3x3.metal") # TODO: insert constants
  ft_prg = MetalProgram("filter_transform", src)
  conv = MetalProgram("conv", src)
  # prog()
