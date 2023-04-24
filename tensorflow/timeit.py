import numpy as np
from time import perf_counter
import tensorflow as tf

def timeit(f, runs=100, timeout=120):
    
    times = []
    total_time = 0

    for r in range(runs):

        if total_time >= timeout:
            break
        
        s = tf.timestamp()
        
        f()

        dt = tf.timestamp() - s
        times.append(dt)
        total_time += dt

    median_time = np.sort(times)[len(times)//2]

    return median_time
