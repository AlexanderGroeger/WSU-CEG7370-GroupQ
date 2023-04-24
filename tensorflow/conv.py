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

def convolution(matrix):
    return tf.nn.conv2d(matrix, kernel, strides=[1, 1, 1, 1], padding="VALID")

# Test CPU execution time
with tf.device('/CPU:0'):
    cpu_time = timeit(lambda: convolution(input_data))

# Test GPU execution time
with tf.device('/GPU:0'):
    gpu_time = timeit(lambda: convolution(input_data))

print("CPU Time", cpu_time)
print("GPU Time", gpu_time)
print("Speed Up :", cpu_time/gpu_time)
