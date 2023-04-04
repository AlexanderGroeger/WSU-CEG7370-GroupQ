import tensorflow as tf
import time

# create two large matrices
matrix1 = tf.random.normal([1000, 1000])
matrix2 = tf.random.normal([1000, 1000])

# copy matrices to CPU memory
start_time = time.time()
with tf.device('/CPU:0'):
    matrix1_cpu = tf.identity(matrix1)
    matrix2_cpu = tf.identity(matrix2)
end_time = time.time()
cpu_copy_time = end_time - start_time

# copy matrices to GPU memory
start_time = time.time()
with tf.device('/GPU:0'):
    matrix1_gpu = tf.identity(matrix1)
    matrix2_gpu = tf.identity(matrix2)
end_time = time.time()
gpu_copy_time = end_time - start_time

# perform matrix multiplication on CPU
start_time = time.time()
with tf.device('/CPU:0'):
    result_cpu = tf.matmul(matrix1_cpu, matrix2_cpu)
end_time = time.time()
cpu_time = end_time - start_time

# perform matrix multiplication on GPU
start_time = time.time()
with tf.device('/GPU:0'):
    result_gpu = tf.matmul(matrix1_gpu, matrix2_gpu)
end_time = time.time()
gpu_time = end_time - start_time

print("CPU copy time:", cpu_copy_time)
print("GPU copy time:", gpu_copy_time)
print("CPU time:", cpu_time)
print("GPU time:", gpu_time)
print("Speedup:", cpu_time / gpu_time)
