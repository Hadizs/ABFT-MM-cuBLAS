CFLAGS=--gpu-architecture sm_52 -lcurand -lcudart -lcublas -g -O3

all:
	nvcc mmul_1.cu -o mmul_1 $(CFLAGS)
	nvcc mmul_2.cu -o mmul_2 $(CFLAGS)
	nvcc mmul_1_fault.cu -o mmul_1_fault $(CFLAGS)
clean:
	rm -f mmul_1 mmul_2 mmul_1_fault
