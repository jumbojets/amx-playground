from tinygrad.runtime.ops_metal import RawMetalBuffer, MetalProgram
from tinygrad.helpers import dtypes
import numpy as np
from string import Template

N, HW, C, F = 1, 512, 16, 1
LID = 2

nims = np.random.default_rng().standard_normal(size=(N,C,HW,HW), dtype=np.float32)
nfs = np.random.default_rng().standard_normal(size=(F,C,3,3), dtype=np.float32)

ims = RawMetalBuffer.fromCPU(nims)
fs = RawMetalBuffer.fromCPU(nfs)
fts = RawMetalBuffer(F*C*4*4, dtypes.float32)
out = RawMetalBuffer((HW-2)*(HW-2), dtypes.float32)

src = Template(open("conv2d_wino_2x2_3x3.metal", "r").read()).substitute(N=N,HW=HW,C=C,F=F)

ft_prg = MetalProgram("filter_transform", src)
conv_prg = MetalProgram("conv", src)

ft_prg([HW,1,1], [C,1,1], fts, fs, wait=True)

print(fs.toCPU())
print(fts.toCPU())

conv_prg([HW//32, HW//(32*LID), 1], [1, LID, 1], out, ims, fts)

print(out.toCPU())
