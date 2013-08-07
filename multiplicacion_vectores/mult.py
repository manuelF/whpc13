import numpy
import pycuda.autoinit
from pycuda import gpuarray
a_cpu = numpy.random.randn(1,10**8).astype(numpy.float32)
b_cpu = numpy.random.randn(1,10**8).astype(numpy.float32)
c_cpu = a_cpu * b_cpu
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = (a_gpu * b_gpu).get()
print c_cpu - c_gpu
