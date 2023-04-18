import os
os.environ["CUPY_ACCELERATORS"] = 'cub'

import numpy as np
import cupy as cp
import cv2
from timeit import timeit
from runit import runit

def dataFunction(size):

    a = np.random.uniform(0,1,size=(size,size,3)).astype(np.float32)
    ga = cp.array(a)

    c = np.array([0.299,0.587,0.114],dtype=np.float32)
    gc = cp.array(c)
    
    return (a,c), (ga,gc)

def cpuFunction(data):
    a, c = data
    return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)

def gpuFunction(data):
    ga, gc = data
    return cp.dot(ga,gc)

def gpuMemFunction(data):
    a, c = data
    return cp.dot(cp.array(a),cp.array(c)).get()

runit(dataFunction, cpuFunction, gpuFunction, gpuMemFunction)
