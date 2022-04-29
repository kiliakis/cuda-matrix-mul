# CUDA_PREFIX = /usr/local/cuda-10.0

CC = nvcc
GPU_CFLAGS = -O3 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70
LIBS = -lm -lcuda

all: main

main: main.cu
	$(CC) $(GPU_CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f main
