all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 latency-meansurement.cu -o measurement

clean:
	rm -f measurement
