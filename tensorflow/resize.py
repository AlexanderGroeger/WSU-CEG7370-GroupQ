import time
import numpy as np
import tensorflow as tf

# Set random seed for reproducibility
tf.random.set_seed(42)

# Generate a random image
image_size = np.random.randint(512, 512)  # Random image size between 1024 and 4096
image = np.random.rand(image_size, image_size, 3).astype(np.float32)

# Define the new image size for resizing
new_image_size = np.random.randint(512, 512)  # Random new image size between 1024 and 4096

# Define the bicubic interpolation function
def bicubic_interpolation(image, new_size):
    # Expand the dimensions of the input image
    image = tf.expand_dims(image, axis=0)
                    
    # Perform bicubic interpolation
    resized_image = tf.image.resize(image, [new_size, new_size], method='bicubic')
                            
    # Remove the first dimension of the output tensor
    resized_image = tf.squeeze(resized_image, axis=0)
                                    
    return resized_image

# Define the bilinear interpolation function
def bilinear_interpolation(image, new_size):
    # Expand the dimensions of the input image
    image = tf.expand_dims(image, axis=0)
                                                
    # Perform bilinear interpolation
    resized_image = tf.image.resize(image, [new_size, new_size], method='bilinear')
                                                        
    # Remove the first dimension of the output tensor
    resized_image = tf.squeeze(resized_image, axis=0)
                                                                
    return resized_image

# Perform bicubic interpolation on the CPU
start_time_cpu = time.time()
resized_image_cpu_bicubic = bicubic_interpolation(image, new_image_size)
end_time_cpu = time.time()

# Calculate the CPU execution time for bicubic interpolation
cpu_time_bicubic = end_time_cpu - start_time_cpu
print(f"CPU execution time for bicubic interpolation: {cpu_time_bicubic:.6f} seconds")

# Perform bicubic interpolation on the GPU
with tf.device('/GPU:0'):
    start_time_gpu = time.time()
    resized_image_gpu_bicubic = bicubic_interpolation(image, new_image_size)
    end_time_gpu = time.time()
# Calculate the GPU execution time for bicubic interpolation
gpu_time_bicubic = end_time_gpu - start_time_gpu
print(f"GPU execution time for bicubic interpolation: {gpu_time_bicubic:.6f} seconds")

# Calculate the speedup for bicubic interpolation
speedup_bicubic = cpu_time_bicubic / gpu_time_bicubic
print(f"Speedup for bicubic interpolation: {speedup_bicubic:.6f}")

# Perform bilinear interpolation on the CPU
start_time_cpu = time.time()
resized_image_cpu_bilinear = bilinear_interpolation(image, new_image_size)
end_time_cpu = time.time()

# Calculate the CPU execution time for bilinear interpolation
cpu_time_bilinear = end_time_cpu - start_time_cpu
print(f"CPU execution time for bilinear interpolation: {cpu_time_bilinear:.6f} seconds")

# Perform bilinear interpolation on the GPU
with tf.device('/GPU:0'):
    start_time_gpu = time.time()
    resized_image_gpu_bilinear = bilinear_interpolation(image, new_image_size)
    end_time_gpu = time.time()

# Calculate the GPU execution time for bilinear interpolation
gpu_time_bilinear = end_time_gpu - start_time_gpu
print(f"GPU execution time for bilinear interpolation: {gpu_time_bilinear:.6f} seconds")

# Calculate the speedup for bilinear interpolation
speedup_bilinear = cpu_time_bilinear / gpu_time_bilinear
print(f"Speedup for bilinear interpolation: {speedup_bilinear:.6f}")
