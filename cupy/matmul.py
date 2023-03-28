import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from timeit import timeit

n=1000
print(f"Size: {n}x{n}")

a = np.random.uniform(0,1,size=(n,n))
b = np.random.uniform(0,1,size=(n,n))
ga = cp.array(a)
gb = cp.array(b)

ta = timeit("small_numpy",lambda: np.matmul(a,b),timeout=30)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: cp.matmul(ga,gb),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: cp.matmul(cp.array(a),cp.array(b)).get(),timeout=20,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del gb
del a
del b


N=10000
print(f"Size: {N}x{N}")


A = np.random.uniform(0,1,size=(N,N))
B = np.random.uniform(0,1,size=(N,N))
gA = cp.array(A)
gB = cp.array(B)

tA = timeit("big_numpy", lambda: np.matmul(A,B),timeout=300)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy", lambda: cp.matmul(gA,gB),timeout=30,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("small_cupy_memory",lambda: cp.matmul(cp.array(A),cp.array(B)).get(),timeout=60,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
