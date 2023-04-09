import numpy as np
import torch
import time

# Define input image and kernel sizes
image_size = (512, 512)
kernel_size = (5, 5)

# Create random image and kernel arrays
image = np.random.rand(*image_size)
kernel = np.random.rand(*kernel_size)

# Implement 2D convolution on CPU
start_time = time.time()
output_cpu = np.zeros_like(image)
for i in range(image.shape[0] - kernel.shape[0] + 1):
    for j in range(image.shape[1] - kernel.shape[1] + 1):
        output_cpu[i, j] = np.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
cpu_time = time.time() - start_time

# Convert arrays to PyTorch tensors and move to GPU
image_gpu = torch.from_numpy(image).cuda()
kernel_gpu = torch.from_numpy(kernel).cuda()

# Implement 2D convolution on GPU
start_time = time.time()
output_gpu = torch.nn.functional.conv2d(
    image_gpu.unsqueeze(0).unsqueeze(0), 
    kernel_gpu.unsqueeze(0).unsqueeze(0)
).squeeze().cpu().numpy()
gpu_time = time.time() - start_time

# Print results
print("CPU time: {:.5f} seconds".format(cpu_time))
print("GPU time: {:.5f} seconds".format(gpu_time))
print("Speedup: {:.5f}x".format(cpu_time / gpu_time))
