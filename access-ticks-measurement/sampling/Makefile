all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 kernel.cu -o kernel
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 kernel-p2p.cu -o kernel-p2p
clean:
	rm -f kernel
