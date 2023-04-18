import tensorflow as tf
import numpy as np
import time
from timeit import timeit

# Define the matrix dimensions
#matrix_size = 512
matrix_size = 4096
kernel_size = 3
num_channels = 1

# Create a random matrix
matrix = tf.random.normal(shape=(1, matrix_size, matrix_size, num_channels), dtype=tf.float32)

# Create a random kernel
kernel = tf.random.normal(shape=(kernel_size, kernel_size, num_channels, num_channels), dtype=tf.float32)

# Define the CPU and GPU devices
cpu_device = '/device:CPU:0'
gpu_device = '/device:GPU:0'

# Define the TensorFlow function
@tf.function
def convolution():
#def convolution(matrix, kernel):
    # Apply the convolution using conv2d
    result = tf.nn.conv2d(matrix, kernel, strides=[1, 1, 1, 1], padding='SAME')
    return result

# Run the computation on the CPU and measure the time taken
with tf.device(cpu_device):
    '''start_time = time.time()
    result_cpu = convolution(matrix, kernel)
    time_cpu = time.time() - start_time'''
    cpu_time = timeit(convolution)
    print("CPU time:", cpu_time)

# Run the computation on the GPU and measure the time taken
with tf.device(gpu_device):
    '''start_time = time.time()
    result_gpu = convolution(matrix, kernel)
    time_gpu = time.time() - start_time'''
    cpu_time = timeit(convolution)
    print("GPU time:", gpu_time)

# Compute the speedup achieved by running the computation on the GPU
speedup = time_cpu / time_gpu
print("Speedup:", speedup)
