import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve2d
import cv2
from timeit import timeit

n=1024
print(f"Size: {n}x{n}")

a = np.random.uniform(0,1,size=(n,n))
k = np.random.uniform(0,1,size=(5,5))
ga = cp.array(a)
gk = cp.array(k)

ta = timeit("small_opencv",lambda: cv2.filter2D(a, ddepth=-1, kernel=k),timeout=20)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: convolve2d(ga,gk),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: convolve2d(cp.array(a),cp.array(k)).get(),timeout=10,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del a


N=8192
print(f"Size: {N}x{N}")

A = np.random.uniform(0,1,size=(N,N))
gA = cp.array(A)

tA = timeit("big_opencv",lambda: cv2.filter2D(A, ddepth=-1, kernel=k),timeout=60)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy",lambda: convolve2d(gA,gk),timeout=30,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("big_cupy_memory",lambda: convolve2d(cp.array(A),cp.array(k)).get(),timeout=30,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
