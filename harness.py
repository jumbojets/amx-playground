from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram
from tinygrad.helpers import dtypes
import numpy as np
from string import Template

N = 1
HW = 512
C = 16
F = 1

nims = np.random.default_rng().standard_normal(size=(N,C,HW,HW), dtype=np.float32)
nfs = np.random.default_rng().standard_normal(size=(F,C,4,4), dtype=np.float32)

ims = RawMetalBuffer.fromCPU(nims)
fs = RawMetalBuffer.fromCPU(nfs)
out = RawMetalBuffer((HW-2)*(HW-2), dtypes.float32)

src = Template(open("conv2d_wino_2x2_3x3.metal", "r").read()).substitute(N=N,HW=HW,C=C,F=F)

ft_prg = MetalProgram("filter_transform", src)
conv_prg = MetalProgram("conv", src)
