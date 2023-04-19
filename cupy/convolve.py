import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.signal import convolve2d
import cv2
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size))
    k = np.random.uniform(0,1,size=(5,5))

    ga = cp.array(a)
    gk = cp.array(k)

    return (a,c), (ga,gc)

def cpuFunction(data):
    a, k = data
    return cv2.filter2D(a, ddepth=-1, kernel=k)

def gpuFunction(data):
    ga, gc = data
    return convolve2d(ga,gk)

def gpuMemFunction(data):
    a, c = data
    return convolve2d(cp.array(a),cp.array(k)).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
