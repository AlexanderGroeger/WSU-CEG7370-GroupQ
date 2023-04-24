import tensorflow as tf
from timeit import timeit

sizes = [512, 2048, 8192]

for size in sizes:
    # create two large matrices
    matrix1 = tf.random.normal([size, size])
    matrix2 = tf.random.normal([size, size])

    # define function to perform matrix multiplication on GPU
    @tf.function
    def gpu_multiply(a, b):
        with tf.device('/GPU:0'):
            return tf.matmul(a, b)

    # copy matrices to CPU memory
    cpu_copy_time = timeit(lambda: tf.identity(matrix1))

    # copy matrices to GPU memory
    gpu_copy_time = timeit(lambda: tf.identity(matrix1).gpu())

    # perform matrix multiplication on CPU
    cpu_time = timeit(lambda: tf.matmul(matrix1, matrix2))

    # perform matrix multiplication on GPU
    gpu_time = timeit(lambda: tf.matmul(matrix1.gpu(), matrix2.gpu()))

    operation_speedup = cpu_time / gpu_time

    print(f"Matrix size: {size}")
    print(f'CPU copy time: {cpu_copy_time:.3e}')
    print(f'GPU copy time: {gpu_copy_time:.3e}')
    print(f'CPU time: {cpu_time:.3e}')
    print(f'GPU time: {gpu_time:.3e}')
    print(f'Operation speedup: {operation_speedup:.2f}')
    print()
