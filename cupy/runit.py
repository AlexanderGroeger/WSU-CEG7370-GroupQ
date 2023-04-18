from timeit import timeit

sizes = (512, 2048, 8192)

def runit(dataFunction, cpuFunction, gpuFunction, gpuMemoryFunction):

    for size in sizes:
        
        print("size: {size}x{size}")

        cpuData, gpuData = dataFunction(size)
        
        cpuTime = timeit(lambda: cpuFunction(cpuData))
        print(f"cpu: {cpuTime:.3e}")
    
        gpuTime = timeit(lambda: gpuFunction(gpuData), gsync=True)
        print(f"gpu: {gpuTime:.3e}")

        gpuMemTime = timeit(lambda: gpuMemoryFunction(cpuData), gsync=True)
        print(f"gpu+mem: {gpuMemTime:.3e}")
    
