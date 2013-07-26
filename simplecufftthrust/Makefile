# Paths & Includes
CUDA_INSTALL_PATH       = /opt/cuda-5.0
CUDA_INSTALL_PATH		= ''
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib64
INCLUDES += -I$(CUDA_INSTALL_PATH)/include -I./include/ -I../common -I/opt/fftw/3.3.3/include

# Parameters
TAMANIO			= 4194304
PARAMETROS		= -DTAMANIO=$(TAMANIO) -DIMPRIMIR 

# Flags
LIBS                    = -L$(CUDA_LIB_PATH)
LDFLAGS                 = -lrt -lm -lcudart -lcufft 
NVCC_FLAGS              = -O2 --use_fast_math   
#OMP_FLAGS        	= -O2 -fopenmp -DTHRUST_HOST_BACKEND=THRUST_HOST_BACKEND_OMP -lgomp -lfftw3 -lm -L/opt/fftw-3.3.3/lib 
FFTW_FLAGS		= -O2 -lfftw3 -lm -L/opt/fftw/3.3.3/lib 
MULTIFFTW_FLAGS		= -O2 -fopenmp -lgomp -lfftw3_threads -lpthread -lfftw3 -lm -L/opt/fftw/3.3.3/lib -lrt -DOMP
NVCC                    = $(CUDA_INSTALL_PATH)nvcc -arch=sm_20
GCC			= icpc
LINKER			= $(NVCC)

# Rules
all: clean simple_cufft simple_fftw simple_fftw_threads

simple_cufft: simple_cufft_thrust.cu
	$(LINKER) $(NVCC_FLAGS) -o simple_cufft simple_cufft_thrust.cu $(PARAMETROS) $(LDFLAGS) $(INCLUDES) $(LIBS)
	  
simple_fftw: simple_fftw_thrust.cpp
	$(GCC) -o simple_fftw simple_fftw_thrust.cpp $(PARAMETROS) $(FFTW_FLAGS) $(INCLUDES)	 

simple_fftw_threads: simple_fftw_thrust.cpp
	$(GCC) -o simple_fftw_threads simple_fftw_thrust.cpp -DFFTWTHREADS $(PARAMETROS) $(MULTIFFTW_FLAGS) $(INCLUDES)	 

clean:
	rm -f simple_cufft simple_fftw simple_fftw_threads *.o *.linkinfo cuda_profile_?.log
