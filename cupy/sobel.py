import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from scipy.ndimage import sobel as cpuSobel
from cupyx.scipy.ndimage import sobel as gpuSobel
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.randint(0,255,size=(size,size),dtype=np.uint8)
    ga = cp.array(a)

    return a, ga

cpuFunction = cpuSobel
gpuFunction = gpuSobel

def gpuMemFunction(data):
    return gpuSobel(cp.array(data)).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
