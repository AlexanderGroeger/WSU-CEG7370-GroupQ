import tensorflow as tf
import numpy as np
import time

# Generate random input data
#input_data = np.random.rand(1, 512, 512).astype(np.float32)
#input_data = np.random.rand(1, 4096, 4096).astype(np.float32)
input_data = np.random.rand(1, 8192, 8192).astype(np.float32)

# Define the rotation function
def rotate_matrix(input_matrix):
    angle = np.radians(45)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_matrix = tf.nn.conv2d(np.expand_dims(input_matrix, axis=-1), tf.constant(rotation_matrix, dtype=tf.float32, shape=(2, 2, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
    return rotated_matrix

# Define the model
def model(input_data):
    rotated_output = rotate_matrix(input_data)
    return rotated_output

# Define the CPU and GPU sessions
with tf.device('/cpu:0'): # set device to CPU
    input_tensor = tf.constant(input_data)
    output_tensor = model(input_tensor)
    start_time_cpu = time.time()
    rotated_data_cpu = output_tensor.numpy()
    end_time_cpu = time.time()

with tf.device('/device:GPU:0'): # set device  to GPU
    input_tensor = tf.constant(input_data)
    output_tensor = model(input_tensor)
    start_time_gpu = time.time()
    rotated_data_gpu = output_tensor.numpy()
    end_time_gpu = time.time()

# Calculate and print the speedup
cpu_time = end_time_cpu - start_time_cpu
gpu_time = end_time_gpu - start_time_gpu
speedup = cpu_time / gpu_time
print("CPU time:", cpu_time)
print("GPU time:", gpu_time)
print("Speedup:", speedup)
