import tensorflow as tf
import numpy as np
import time
from timeit import timeit
from PIL import Image
import sys


if len(sys.argv) > 1:
     n = sys.argv[1]
     #Generate random input data
     input_data = np.random.rand(1, int(n), int(n)).astype(np.float32)
     print("Size = ", n, "x", n)
else:
    input_image = Image.open("s-l1600.jpg") # Load input image
    input_data = np.array(input_image).astype(np.float32) # Convert input image to numpy array
    input_data = np.expand_dims(input_data, axis=0) # Add batch dimension

# Define the rotation function
def rotate_matrix(input_matrix):
     angle = np.radians(90)
     #rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
     #rotated_matrix = tf.nn.conv2d(np.expand_dims(input_matrix, axis=-1), tf.constant(rotation_matrix, dtype=tf.float32, shape=(2, 2, 1, 1)), strides=[1, 1, 1, 1], padding='SAME')
     rotated_matrix = tf.image.rot90(input_matrix, k=int(angle // (np.pi / 2)))
     return rotated_matrix


# Define the CPU and GPU sessions
def run_on_cpu():
    with tf.device('/cpu:0'): # set device to CPU
        input_tensor = tf.constant(input_data)
        output_tensor = rotate_matrix(input_tensor)
        rotated_data_cpu = output_tensor.numpy()
        #rotated_image_cpu = Image.fromarray(rotated_data_cpu[0, 0, :, :, 0].astype(np.uint8)) # Convert rotated data to image
        if(len(sys.argv) < 2):
            rotated_image_cpu = Image.fromarray(rotated_data_cpu[0].astype(np.uint8)) # Convert rotated data to image
            rotated_image_cpu.save("output_image_cpu.jpg") # Save output image from CPU

def run_on_gpu():
    with tf.device('/device:GPU:0'): # set device to GPU
        input_tensor = tf.constant(input_data)
        output_tensor = rotate_matrix(input_tensor)
        rotated_data_gpu = output_tensor.numpy()
        #rotated_image_gpu = Image.fromarray(rotated_data_gpu[0, 0, :, :, 0].astype(np.uint8)) # Convert rotated data to image
        if(len(sys.argv) < 2):
            rotated_image_gpu = Image.fromarray(rotated_data_gpu[0].astype(np.uint8)) # Convert rotated data to image
            rotated_image_gpu.save("output_image_gpu.jpg") # Save output image from CPU


# Calculate and print the speedup
cpu_time = timeit(run_on_cpu) # Measure CPU time
gpu_time = timeit(run_on_gpu) # Measure GPU time
speedup = cpu_time / gpu_time
print("CPU time:", cpu_time)
print("GPU time:", gpu_time)
print("Speedup:", speedup)
