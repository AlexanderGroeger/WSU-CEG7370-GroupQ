import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from scipy.ndimage inport sobel as cpuSobel
from cupyx.scipy.ndimage import gpuSobel
from timeit import timeit

n=1024
print(f"Size: {n}x{n}")

a = np.random.randint(0,255,size=(n,n),dtype=np.uint8)
ga = cp.array(a)

ta = timeit("small_opencv",lambda: cpuSobel(a),timeout=20)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: gpuSobel(ga),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: gpuSobel(cp.array(a)).get(),timeout=10,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del a


N=8192
print(f"Size: {N}x{N}")

A = np.random.randint(0,255,size=(N,N),dtype=np.uint8)
gA = cp.array(A)

tA = timeit("big_opencv",lambda: cpuSobel(A),timeout=60)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy",lambda: gpuSobel(gA),timeout=30,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("big_cupy_memory",lambda: gpuSobel(cp.array(gA)).get(),timeout=30,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
