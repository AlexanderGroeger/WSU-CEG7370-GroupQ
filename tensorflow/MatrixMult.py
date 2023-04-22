import tensorflow as tf
from timeit import timeit

def cpu_multiply(matrix1, matrix2):
    return tf.matmul(matrix1, matrix2)

def gpu_multiply(matrix1, matrix2):
    return tf.matmul(matrix1, matrix2)

sizes = [512, 2048, 8192]

for size in sizes:
    print(f"Size: {size}x{size}")

    # create two large matrices
    matrix1 = tf.random.normal([size, size])
    matrix2 = tf.random.normal([size, size])

    # copy matrices to CPU memory
    cpu_copy_time = timeit(lambda: tf.identity(matrix1), number=1)
    cpu_copy_time += timeit(lambda: tf.identity(matrix2), number=1)

    # copy matrices to GPU memory
    gpu_copy_time = timeit(lambda: tf.identity(matrix1).numpy(), number=1)
    gpu_copy_time += timeit(lambda: tf.identity(matrix2).numpy(), number=1)

    # perform matrix multiplication on CPU
    cpu_time = timeit(lambda: cpu_multiply(matrix1, matrix2), number=1)

    # perform matrix multiplication on GPU
    gpu_time = timeit(lambda: timeit(lambda: gpu_multiply(matrix1, matrix2), number=1), number=1)

    operation_speedup = cpu_time / gpu_time
    gpu_memory_usage = tf.config.experimental.get_memory_usage('GPU:0') / 1024 / 1024 / 1024
    gpu_memory_usage = abs(gpu_memory_usage) # memory usage can be negative in some cases
    gpu_mem_speedup = (cpu_time + cpu_copy_time) / (gpu_time + gpu_copy_time + gpu_memory_usage)
    
    print(f'CPU copy time: {cpu_copy_time:.3e}')
    print(f'GPU copy time: {gpu_copy_time:.3e}')
    print(f'CPU time: {cpu_time:.3e}')
    print(f'GPU time: {gpu_time:.3e}')
    print(f'Operation speedup: {operation_speedup:.2f}')
    print(f'GPU memory usage: {gpu_memory_usage:.3f} GB')
    print(f'GPU memory speedup: {gpu_mem_speedup:.2f}')
