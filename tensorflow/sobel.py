import tensorflow as tf
import numpy as np
import time
from timeit import timeit

# Define the matrix dimensions
image_size = 512
#image_size = 2048
#image_size = 8192

# Create a random image
image = tf.random.normal(shape=(1, image_size, image_size, 3), dtype=tf.float32)

# Define the Sobel edge detection function
@tf.function
def sobel_edge_detection():
    # Apply the Sobel edge detection using tf.image.sobel_edges
    gradients = tf.image.sobel_edges(image)
    return gradients

# Run the Sobel edge detection on the CPU and measure the time taken
with tf.device('/device:CPU:0'):
    cpu_time = timeit(sobel_edge_detection)
    print("CPU time:", cpu_time)

# Run the Sobel edge detection on the GPU and measure the time taken
with tf.device('/device:GPU:0'):
    gpu_time = timeit(sobel_edge_detection)
    print("GPU time:", gpu_time)

# Compute the speedup achieved by running the Sobel edge detection on the GPU
speedup = cpu_time / gpu_time
print("Speedup:", speedup)

