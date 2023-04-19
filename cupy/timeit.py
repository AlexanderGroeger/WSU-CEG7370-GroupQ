import numpy as np
import cupy as cp
from time import time

def timeit(f, gsync=False, runs=100):

    times = []

    for i in range(runs):
        s = time()
        f()
        if gsync:
            cp.cuda.Stream.null.synchronize()
        dt = time() - s
        times.append(dt)
    
    median_time = np.sort(times)[len(times)//2]

    return median_time


def timeit_old(title,f, timeout=10, gsync=False):
    
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

    return median_time
