import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import rotate
import cv2
from timeit import timeit

angle=45

n=512
print(f"Size: {n}x{n}")

a = np.random.uniform(0,1,size=(n,n,3))
ga = cp.array(a)

ta = timeit("small_opencv",lambda: cv2.warpAffine(a,cv2.getRotationMatrix2D((n//2,n//2),angle,1),dsize=(n,n)),timeout=20)
print(f"cpu time: {ta:.4e}")
tb = timeit("small_cupy",lambda: rotate(ga,angle,reshape=False,mode='opencv'),timeout=10,gsync=True)
print(f"gpu time: {tb:.4e}")
print(f"operation speedup: {ta/tb:.2f}")
tc = timeit("small_cupy_memory",lambda: rotate(cp.array(a),angle,reshape=False,mode='opencv').get(),timeout=10,gsync=True)
print(f"gpu+mem time: {tc:.4e}")
print(f"total speedup: {ta/tc:.2f}")

del ga
del a


N=2048
print(f"Size: {N}x{N}")

A = np.random.uniform(0,1,size=(N,N,3))
gA = cp.array(A)

tA = timeit("big_opencv",lambda: cv2.warpAffine(A,cv2.getRotationMatrix2D((N//2,N//2),angle,1),dsize=(N,N)),timeout=20)
print(f"cpu time: {tA:.4e}")
tB = timeit("big_cupy",lambda: rotate(gA,angle,reshape=False,mode='opencv'),timeout=10,gsync=True)
print(f"gpu time: {tB:.4e}")
print(f"operation speedup: {tA/tB:.2f}")
tC = timeit("big_cupy_memory",lambda: rotate(cp.array(A),angle,reshape=False,mode='opencv').get(),timeout=10,gsync=True)
print(f"gpu+mem time: {tC:.4e}")
print(f"total speedup: {tA/tC:.2f}")
