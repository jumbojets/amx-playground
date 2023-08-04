from conv2d_wino import conv2d_wino
import numpy as np
import torch
import torch.nn.functional as torchf
from tinygrad.runtime.ops_metal import RawMetalBuffer
from bench import sample

cases = [(1,128,16,1), (1,32,1,1), (1,32,1,2), (16,16,16,16), (32,128,4,512)]

for args in cases: sample(*args)
