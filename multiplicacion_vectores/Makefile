all: simple_cuda simple_omp

simple_omp: main.cu
	cp main.cu main.cpp; g++ main.cpp -O3 -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP -lgomp -DOMP -I/opt/cuda-5.0/include -o omp.out

simple_cuda: main.cu
	nvcc -arch=sm_20 -O3 main.cu -o ./cuda.out

clean:
	rm *.out main.cpp

