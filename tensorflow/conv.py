import tensorflow as tf
import time
from timeit import timeit
import sys
import numpy as np

# Generate random input data
n = int(sys.argv[1])
print("Size: ",n,"X",n)
input_data = tf.random.normal([1, n, n, 1])

conv_layer = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')

# Test CPU execution time
def run_on_cpu():
    with tf.device('/CPU:0'):
        output_data_cpu = conv_layer(input_data)

# Test GPU execution time
def run_on_gpu():
    with tf.device('/GPU:0'):
        output_data_gpu = conv_layer(input_data)

cpu_time = timeit(run_on_cpu)
gpu_time = timeit(run_on_gpu)
print("CPU Time", cpu_time)
print("GPU Time", gpu_time)
print("Speed Up :", cpu_time/gpu_time)
