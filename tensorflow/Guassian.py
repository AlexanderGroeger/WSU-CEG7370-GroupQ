import tensorflow as tf
import time

# Define the function to be evaluated (Gaussian)
def gaussian(x, mu, sigma):
    return tf.exp(-(x - mu)**2 / (2 * sigma**2)) / tf.sqrt(2 * tf.constant(3.1415) * sigma**2)

# Define the size of the data
n = 10000000

# Generate the data
data = tf.random.normal([n], mean=0.0, stddev=1.0)

# Create a TensorFlow session for CPU computation
with tf.device('/CPU:0'):
    # Evaluate the function on CPU
    start_time = time.time()
    gaussian_cpu = gaussian(data, 0.0, 1.0)
    end_time = time.time()

# Calculate the time taken for CPU computation
cpu_time = end_time - start_time

# Create a TensorFlow session for GPU computation
with tf.device('/GPU:0'):
    # Evaluate the function on GPU
    start_time = time.time()
    gaussian_gpu = gaussian(data, 0.0, 1.0)
    end_time = time.time()

# Calculate the time taken for GPU computation
gpu_time = end_time - start_time

# Calculate the speedup
speedup = cpu_time / gpu_time

# Print the results
print("CPU time",cpu_time)
print("GPU time:",gpu_time)
print("Speedup: ",speedup)