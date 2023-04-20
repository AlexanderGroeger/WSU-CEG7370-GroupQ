import tensorflow as tf
import numpy as np
import time
from timeit import timeit


# Define the matrix dimensions
image_size = 512
#image_size = 2048
#image_size = 8192

# Generate a random 512x512 matrix
input_data = tf.random.normal([512,512])


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

# Create a function for applying the filter to the input matrix
def apply_sobel_filter(input_data):
    # Reshape the input matrix into a 4D tensor (batch_size=1, height=512, width=512, channels=1)
    input_data = tf.reshape(input_data, [1, 512,512, 1])

# Run the Sobel edge detection on the GPU and measure the time taken
with tf.device('/device:GPU:0'):
    gpu_time = timeit(sobel_edge_detection)
    print("GPU time:", gpu_time)


# Compute the speedup achieved by running the Sobel edge detection on the GPU
speedup = cpu_time / gpu_time
print("Speedup:", speedup)

