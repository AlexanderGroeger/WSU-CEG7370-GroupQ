import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
import cv2
from timeit import timeit

n=512
print(f"Size: {n}x{n}")

a = np.random.randint(0,15,size=(n,n),dtype=np.uint8)
b = np.random.randint(0,15,size=(n,n),dtype=np.uint8)
ga = cp.array(a)
gb = cp.array(b)

ta = timeit("small_numpy",lambda: np.matmul(a,b),timeout=30)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: cp.matmul(ga,gb),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: cp.matmul(cp.array(a),cp.array(b)),timeout=20,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del gb
del a
del b


N=4096
print(f"Size: {N}x{N}")

A = np.random.randint(0,15,size=(N,N),dtype=np.uint8)
B = np.random.randint(0,15,size=(N,N),dtype=np.uint8)
gA = cp.array(A)
gB = cp.array(B)

tA = timeit("big_numpy", lambda: np.matmul(A,B),timeout=600)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy", lambda: cp.matmul(gA,gB),timeout=30,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("small_cupy_memory",lambda: cp.matmul(cp.array(A),cp.array(B)),timeout=60,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
