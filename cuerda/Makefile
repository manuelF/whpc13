# Paths & Includes
CUDA_INSTALL_PATH       = /opt/cuda-5.0
CUDA_LIB_PATH           = $(CUDA_INSTALL_PATH)/lib64
INCLUDES += -I$(CUDA_INSTALL_PATH)/include -I./include/ -I../common  

# Flags
LIBS                    = -L$(CUDA_LIB_PATH) 
LDFLAGS                 = -lrt -lm -lcudart -lcufft 
NVCC_FLAGS              = -O2 --use_fast_math

OMP_FLAGS               = -O2 -fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP -lgomp -DOMP
TBB_FLAGS               = -O2 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -ltbb
#TBB_FLAGS               = -O2 -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -ltbb -L$(TBB_PATH)/lib

TAMANIO			= 65536
TIEMPORUN		= 10000
PARAMETERS		= -DTAMANIO=$(TAMANIO) -DTIEMPORUN=$(TIEMPORUN) #-DFINITETEMPERATURE -DPRINTCONFS


NVCC                    = $(CUDA_INSTALL_PATH)/bin/nvcc -arch=sm_20
GCC			= g++
ICC			= /opt/intel/bin/icc
LINKER			= $(NVCC)


# Rules
all: clean hacer qew_CUDA qew_OMP

qew_CUDA: qew_minimal.cu
	$(NVCC) $(NVCC_FLAGS) -o qew_CUDA qew_minimal.cu $(LDFLAGS) $(INCLUDES) $(LIBS) $(PARAMETERS)
	  
qew_OMP: qew_minimal.cpp 
	$(GCC) -o qew_OMP qew_minimal.cpp $(OMP_FLAGS) $(INCLUDES) $(PARAMETERS)
				 
clean:
	rm -f $(BIN) *.o *.linkinfo cuda_profile_?.log qew_CUDA qew_OMP qew_minimal.cpp
                                                     
hacer: 
	cp qew_minimal.cu qew_minimal.cpp
