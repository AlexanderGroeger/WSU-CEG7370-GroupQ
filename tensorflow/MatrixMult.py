import tensorflow as tf
import sys
from timeit import timeit

if len(sys.argv) >1 :
    matrix_size = sys.argv[1]
    matrix1 = tf.random.normal(shape=(int(matrix_size), int(matrix_size)))
    matrix2 = tf.random.normal(shape=(int(matrix_size), int(matrix_size)))
    print("Size: ", matrix_size, 'X', matrix_size)

else:
    printf("Please pass valid paraeter with the script")
    exit(0)

def matrixMultiplication(matrix1, matrix2):
    return tf.matmul(matrix1, matrix2)


with tf.device('/CPU:0'):
    cpu_time = timeit(lambda: matrixMultiplication(matrix1, matrix2))


with tf.device('/GPU:0'):
    gpu_time = timeit(lambda: matrixMultiplication(matrix1, matrix2))


print("CPU time: ", cpu_time, " seconds")
print("GPU time: ", gpu_time," seconds")
speedup = cpu_time / gpu_time
