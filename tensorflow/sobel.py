import tensorflow as tf
import time

# Generate a random 512x512 matrix
input_data = tf.random.normal([512,512])

# Create the Sobel filter
sobel_filter = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
sobel_filter = tf.reshape(sobel_filter, [3, 3, 1, 1])


# Create a function for applying the filter to the input matrix
def apply_sobel_filter(input_data):
    # Reshape the input matrix into a 4D tensor (batch_size=1, height=512, width=512, channels=1)
    input_data = tf.reshape(input_data, [1, 512,512, 1])

    # Apply the Sobel filter to the input matrix using conv2d
    output_data = tf.nn.conv2d(input_data, sobel_filter, strides=[1, 1, 1, 1], padding='SAME')

    # Reshape the output matrix back to a 2D matrix (height=512, width=512)
    output_data = tf.reshape(output_data,512,512])

    return output_data

# Test the function on CPU
with tf.device('/cpu:0'):
    start_time_cpu = time.time()
    output_cpu = apply_sobel_filter(input_data)
    end_time_cpu = time.time()

# Test the function on GPU
with tf.device('/gpu:0'):
    start_time_gpu = time.time()
    output_gpu = apply_sobel_filter(input_data)
    end_time_gpu = time.time()

# Compute the speedup time
speedup_time = (end_time_cpu - start_time_cpu) / (end_time_gpu - start_time_gpu)

# Print the results
print('Time taken on CPU:', end_time_cpu - start_time_cpu)
print('Time taken on GPU:', end_time_gpu - start_time_gpu)
print('Speedup time:', speedup_time)
