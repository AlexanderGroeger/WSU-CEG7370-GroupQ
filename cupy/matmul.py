import os
os.environ["CUPY_ACCELERATORS"] = 'cub'
OUTDIR = os.environ["CEG7370_HOME"] or "./"
NAME = os.path.basename(__file__).rstrip('.py')

import numpy as np
import cupy as cp
import cv2
from time import time
import matplotlib.pyplot as plt

def timeit(title,f, timeout=10, gsync=False):
    
    times = []
    total_time = 0

    while total_time < timeout:
        s = time()
        f()
        if gsync:
            cp.cuda.Stream.null.synchronize()
        dt = time() - s
        times.append(dt)
        total_time += dt

    median_time = np.sort(times)[len(times)//2]
    print(f"{title}: {median_time}")
    plt.figure(figsize=(16,8))
    plt.title(f"{title}: {median_time:.1e} seconds")
    plt.ylabel("time (us)")
    plt.xlabel("iteration")
    plt.yscale('log')
    plt.plot(times[len(times)//8:])
    plt.savefig(f"{OUTDIR}/{NAME}/{title}.png")
    plt.close()

    
n=512
print(f"Size: {n}x{n}")

a = np.random.randint(0,15,size=(n,n),dtype=np.uint8)
b = np.random.randint(0,15,size=(n,n),dtype=np.uint8)
ga = cp.array(a)
gb = cp.array(b)

timeit("small_numpy",lambda: np.matmul(a,b),timeout=30)
timeit("small_cupy",lambda: cp.matmul(ga,gb),timeout=10,gsync=True)

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

timeit("big_numpy", lambda: np.matmul(A,B),timeout=600)
timeit("big_cupy", lambda: cp.matmul(gA,gB),timeout=30,gsync=True)
