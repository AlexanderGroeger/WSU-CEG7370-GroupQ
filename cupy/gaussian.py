import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter
import cv2
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size)).astype(np.float32)
    ga = cp.array(a)
    
    return a, ga

def cpuFunction(data):
    return cv2.GaussianBlur(data,(3,3),sigmaX=1,sigmaY=1)

def gpuFunction(data):
    return gaussian_filter(data,1)

def gpuMemFunction(data):
    return gaussian_filter(cp.array(data),1).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
