import tensorflow as tf
import time
from timeit import timeit
import sys
import numpy as np

if len(sys.argv) > 1:
     n = sys.argv[1]
     random_data = tf.random.uniform(shape=(512, 512), minval=0.0, maxval=1.0)
     image = tf.image.convert_image_dtype(random_data, tf.float32)
     split_dim_size = random_data.shape[-1]
     split_size = split_dim_size // 3
     remainder = split_dim_size % 3
     split_sizes = [split_size] * 3
     split_sizes[0] += remainder
     r, g, b = tf.split(random_data, num_or_size_splits=split_sizes, axis=-1)
     print("Size = ", n, "x", n)

else:
    image_path = 'image.jpg'
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    r, g, b = tf.split(image, 3, axis=-1)

def histogram_equalization(channel):
    hist = tf.histogram_fixed_width(channel, value_range=(0.0, 1.0), nbins=256)
    cdf = tf.cumsum(hist)
    cdf = cdf / tf.reduce_max(cdf)
    equalized_channel = tf.gather(cdf, tf.cast(channel * 255.0, tf.int32))
    return equalized_channel

def run_on_cpu():
    with tf.device('/CPU:0'):
        r_equalized = histogram_equalization(r)
        g_equalized = histogram_equalization(g)
        b_equalized = histogram_equalization(b)
        image_equalized = tf.concat([r_equalized, g_equalized, b_equalized], axis=-1)
        image_equalized = tf.cast(image_equalized * 255.0, tf.uint8)
        image_equalized = tf.clip_by_value(image_equalized, 0, 255)
        if(len(sys.argv) < 2):
            tf.io.write_file('he_cpu.jpg', tf.image.encode_jpeg(image_equalized))


def run_on_gpu():
    with tf.device('/GPU:0'):
        r_equalized = histogram_equalization(r)
        g_equalized = histogram_equalization(g)
        b_equalized = histogram_equalization(b)
        image_equalized = tf.concat([r_equalized, g_equalized, b_equalized], axis=-1)
        image_equalized = tf.cast(image_equalized * 255.0, tf.uint8)
        image_equalized = tf.clip_by_value(image_equalized, 0, 255)
        if(len(sys.argv) < 2):
            tf.io.write_file('he_gpu.jpg', tf.image.encode_jpeg(image_equalized))


cpu_time = timeit(run_on_cpu)
gpu_time = timeit(run_on_gpu)
print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")
speedup = cpu_time / gpu_time
print("Speedup: ", speedup)
