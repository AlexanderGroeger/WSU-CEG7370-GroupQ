import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from timeit import timeit
import sys

if len(sys.argv) > 1:
     n = sys.argv[1]
     image_dtype = tf.float32
     image = tf.random.uniform(shape=(int(n),int(n)), minval=0, maxval=255, dtype=image_dtype)
     print("Size = ", n, "x", n)

angle = np.deg2rad(45)

def rotateImage(image, angle):
    rotated_image = tfa.image.rotate(image, angle)
    rotated_image = tf.cast(rotated_image, tf.uint8)
    return rotated_image

#To Rotate Actual Image
if (len(sys.argv)<2):
    image_path = 'image.jpg'
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    rotated_image = rotateImage(image, angle)
    tf.io.write_file('rotated_output.jpg', tf.image.encode_jpeg(rotated_image))
    exit(0)


with tf.device('/CPU:0'):
    cpu_time = timeit(lambda: rotateImage(image, angle))

with tf.device('/GPU:0'):
    gpu_time = timeit(lambda: rotateImage(image, angle))

print('CPU Time: ', cpu_time)
print('GPU Time: ', gpu_time)
print('Speed Up: ', cpu_time/gpu_time)
