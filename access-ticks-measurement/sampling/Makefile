all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 kernel.cu -o kernel
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 kernel-p2p.cu -o kernel-p2p
clean:
	rm -f kernel
	rm -f kernel-p2p
