all:
	# nvcc -ccbin=$(CXX) -O3 -arch=sm_80 latency-meansurement.cu -o measurement
	# nvcc -ccbin=$(CXX) -O3 -arch=sm_80 latency-meansurement-p2p.cu -o measurement-p2p
	nvcc -ccbin=$(CXX) -O3 -arch=sm_35 latency-meansurement.cu -o measurement

clean:
	rm -f measurement
