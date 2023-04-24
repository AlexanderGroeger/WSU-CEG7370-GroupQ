import tensorflow as tf
import time
from timeit import timeit
import sys
import numpy as np

# Generate random input data
n = int(sys.argv[1])
print("Size: ",n,"X",n)
input_data = tf.random.normal([1, n, n, 1])

kernel = tf.constant(tf.ones(shape=[1, 5, 5, 1]), dtype=tf.float32)

# Test CPU execution time
with tf.device('/CPU:0'):
    cpu_time = timeit(lambda: tf.nn.conv2d(input_data, kernel, strides=[1, 1, 1, 1], padding="VALID"))

# Test GPU execution time
with tf.device('/GPU:0'):
    gpu_time = timeit(lambda: tf.nn.conv2d(input_data, kernel, strides=[1, 1, 1, 1], padding="VALID"))

print("CPU Time", cpu_time)
print("GPU Time", gpu_time)
print("Speed Up :", cpu_time/gpu_time)
