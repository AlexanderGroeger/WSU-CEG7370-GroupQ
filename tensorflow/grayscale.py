import tensorflow as tf
import numpy as np
import time
from timeit import timeit
from PIL import Image
import sys

if len(sys.argv) > 1:
     n = sys.argv[1]
     random_data = tf.random.normal((512, 512, 3))
     input_data = tf.convert_to_tensor(random_data)
     print("Size = ", n, "x", n)

else:
    input_image = Image.open("s-l1600.jpg") 
    input_data = tf.convert_to_tensor(np.array(input_image))

def rgb2gray(input_data):
    input_data = tf.image.rgb_to_grayscale(input_data)
    input_data = tf.squeeze(input_data, axis=-1)
    input_data = tf.cast(input_data, dtype=tf.uint8)
    gray = input_data.numpy()
    return gray

def run_on_cpu():
    with tf.device('/CPU:0'):
        output_tensor = rgb2gray(input_data)
        if(len(sys.argv) < 2):
            output_image = Image.fromarray(output_tensor, "L")
            output_image.save("gray_cpu.jpg")

def run_on_gpu():
    with tf.device('/GPU:0'):
        output_tensor = rgb2gray(input_data)
        if(len(sys.argv) < 2):
            output_image = Image.fromarray(output_tensor, "L")
            output_image.save("gray_gpu.jpg")


cpu_time = timeit(run_on_cpu)
gpu_time = timeit(run_on_gpu)
 
# Print the CPU and GPU time
print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")

# Calculate the speedup
speedup = cpu_time / gpu_time
print("Speedup: ", speedup)
