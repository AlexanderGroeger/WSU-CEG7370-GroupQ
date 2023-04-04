import tensorflow as tf
import time

# Generate a random 512x512 RGB image
#img = tf.random.normal((512, 512, 3))
img = tf.random.normal((2048, 2048, 3))

# Define a function to convert RGB to grayscale on CPU
def rgb2gray_cpu():
    start_time = time.time()
    with tf.device('/CPU:0'):
        gray_cpu = tf.reduce_sum(img * tf.constant([0.2989, 0.5870, 0.1140]), axis=-1, keepdims=True)
    end_time = time.time()
    return end_time - start_time, gray_cpu

# Define a function to convert RGB to grayscale on GPU
def rgb2gray_gpu():
    start_time = time.time()
    with tf.device('/GPU:0'):
        gray_gpu = tf.reduce_sum(img * tf.constant([0.2989, 0.5870, 0.1140]), axis=-1, keepdims=True)
    end_time = time.time()
    return end_time - start_time, gray_gpu

# Convert RGB to grayscale on CPU and GPU
cpu_time, gray_cpu = rgb2gray_cpu()
gpu_time, gray_gpu = rgb2gray_gpu()

# Print the CPU and GPU time
print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")

# Calculate the speedup
speedup = cpu_time / gpu_time
print("Speedup: ", speedup)
