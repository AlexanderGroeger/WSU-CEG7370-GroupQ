import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from timeit import timeit

n = 1024
print(f"size: {n}x{n}")

a = np.random.uniform(0,1,size=(n,n))
ga = cp.array(a)

ta = timeit("small_numpy",lambda: np.linalg.inv(a),30)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: cp.linalg.inv(ga),20,True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: cp.linalg.inv(cp.array(a)),60,True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del a
del ga

N = 8192
print(f"size: {N}x{N}")

A = np.random.uniform(0,1,size=(N,N))
gA = cp.array(A)

tA = timeit("big_numpy",lambda: np.linalg.inv(A),30)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy",lambda: cp.linalg.inv(gA),20,True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("big_cupy_memory",lambda: cp.linalg.inv(cp.array(A)),60,True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
