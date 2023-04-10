import tensorflow as tf
import numpy as np
import time

# Define the Gaussian kernel
def gaussian_kernel(size, sigma):
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    y = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    x, y = tf.meshgrid(x, y)
    kernel = tf.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= tf.reduce_sum(kernel)
    kernel = tf.reshape(kernel, [size, size, 1, 1]) # reshape the kernel tensor
    return kernel

# Define the Gaussian filter function
def gaussian_filter(image, kernel):
    image = tf.expand_dims(image, axis=0)
    image = tf.expand_dims(image, axis=-1)
    filtered_image = tf.nn.conv2d(image, kernel, strides=1, padding='SAME')
    filtered_image = tf.squeeze(filtered_image)
    return filtered_image

# Set the image size
image_size = 512

# Generate a random image of the specified size
image = np.random.rand(image_size, image_size).astype(np.float32)

# Define the Gaussian kernel parameters
kernel_size = 5
sigma = 1

# Create the Gaussian kernel
kernel = gaussian_kernel(kernel_size, sigma)

# Apply the Gaussian filter to the input image
start_time_cpu = time.time()
filtered_image_cpu = gaussian_filter(image, kernel)
end_time_cpu = time.time()

# Calculate CPU execution time
execution_time_cpu = end_time_cpu - start_time_cpu

# Print CPU execution time
print("CPU execution time: {} seconds".format(execution_time_cpu))

# Apply the Gaussian filter to the input image using the GPU
with tf.device('/GPU:0'):
    start_time_gpu = time.time()
    filtered_image_gpu = gaussian_filter(image, kernel)
    end_time_gpu = time.time()

# Calculate GPU execution time
execution_time_gpu = end_time_gpu - start_time_gpu

# Print GPU execution time and speedup
print("GPU execution time: {} seconds".format(execution_time_gpu))
print("Speedup: {}".format(execution_time_cpu / execution_time_gpu))
