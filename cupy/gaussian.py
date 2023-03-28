import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import cv2
from timeit import timeit

n=1024
print(f"Size: {n}x{n}")

a = np.random.randint(0,255,size=(n,n),dtype=np.uint8)
ga = cp.array(a)

ta = timeit("small_opencv",lambda: cv2.GaussianBlur(a,(3,3),sigmaX=1,sigmaY=1),timeout=20)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: gaussian_filter(ga,1),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: gaussian_filter(cp.array(a),1).get(),timeout=10,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del a


N=8192
print(f"Size: {N}x{N}")

A = np.random.randint(0,255,size=(N,N),dtype=np.uint8)
gA = cp.array(A)

tA = timeit("big_opencv",lambda: cv2.GaussianBlur(A,(3,3),sigmaX=1,sigmaY=1),timeout=60)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy",lambda: gaussian_filter(gA,1),timeout=30,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("big_cupy_memory",lambda: gaussian_filter(cp.array(gA),1).get(),timeout=30,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
