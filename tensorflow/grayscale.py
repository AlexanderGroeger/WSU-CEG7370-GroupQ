import tensorflow as tf
import time
from timeit import timeit

# Generate a random 512x512 RGB image
#img = tf.random.normal((512, 512, 3))
img = tf.random.normal((2048, 2048, 3))

def rgb2gray():
    gray = tf.reduce_sum(img * tf.constant([0.2989, 0.5870, 0.1140]), axis=-1, keepdims=True)
    return gray
    
with tf.device('/CPU:0'):
    cpu_time = timeit(rgb2gray)
    
with tf.device('/GPU:0'):
    gpu_time = timeit(rgb2gray)


# Print the CPU and GPU time
print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")

# Calculate the speedup
speedup = cpu_time / gpu_time
print("Speedup: ", speedup)
