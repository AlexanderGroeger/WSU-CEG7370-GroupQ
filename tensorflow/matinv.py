import tensorflow as tf
import sys
from timeit import timeit

if len(sys.argv) >1 :
    matrix_size = int(sys.argv[1])
    matrix = tf.random.normal((matrix_size, matrix_size))
    print("Size: ", matrix_size, 'X', matrix_size)

else:
    printf("Please pass valid paraeter with the script")
    exit(0)

def matrixInverse(matri):
    return tf.linalg.inv(matrix)

with tf.device('/CPU:0'):
    cpu_time = timeit(lambda: matrixInverse(matrix))

with tf.device('/GPU:0'):
    gpu_time = timeit(lambda: matrixInverse(matrix))


print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")
speedup = cpu_time / gpu_time
print("Speedup: ", speedup)
