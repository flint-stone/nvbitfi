#include <stdio.h> 
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__global__ void access_memory(ulong* memory, ulong* offset, int count){
    int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    ulong *p = memory;
    p += offset[global_id];

    ulong q;
    for(int i=0; i<count; i++){
        q = (ulong)memory + *p;
        p = (ulong*)q; 
    }
    *p = 1;
}

int draw(int limit) { return rand() % limit; }

void swap(uint64_t *a, uint64_t *b) {
    uint64_t temp = *a;
    *a = *b;
    *b = temp;
}

uint64_t getTimeInNSecs() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    uint64_t timeInSec = time.tv_sec * 1e9 + time.tv_nsec;
    return timeInSec;
}

int main(int argc, char **argv){
    uint64_t start, end, kernel_time;
    int len, size;
    unsigned int count;
    size_t grid_size, workgroup_size;

    if (argc != 5) {
        printf("./kernel memory_size num_wg count\n");
        exit(-1);
    }

    size = atoi(argv[1]);
    count = atoi(argv[3]);
    int num_wg = atoi(argv[2]);
    int gpuID = atoi(argv[4]);
    int num_wi_per_wg = 32;

    grid_size = num_wg * num_wi_per_wg;
    workgroup_size = num_wi_per_wg;

    len = size / sizeof(void *);
    int M = pow(2, 24);

    /*P2P*/
    cudaDeviceProp prop[64];
    checkCudaErrors(cudaGetDeviceProperties(&prop[0], 0));
    checkCudaErrors(cudaGetDeviceProperties(&prop[1], 1));
    int p2pCapableGPUs[2]; 
    p2pCapableGPUs[0] = 0;
    p2pCapableGPUs[1] = 1;
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[1], 0));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[0], 0));


    /* Host*/
    uint64_t *memory = new uint64_t[M * sizeof(uint64_t)]; //malloc(M * sizeof(uint64_t));
    uint64_t *indices = new uint64_t[M * sizeof(uint64_t)]; //malloc(M * sizeof(uint64_t));
    for (int i = 0; i < len; i++) {
        indices[i] = i;
    }
    for (int i = 0; i < len - 1; i++) {
        int j = i + draw(len - i);
        if (j != i) {
            swap(&indices[i], &indices[j]);
        }
    }

    for (int i = 1; i < len; i++) {
        memory[indices[i - 1]] = indices[i] * 8;
    }
    memory[indices[len - 1]] = indices[0] * 8;

    /* Data copy */
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    uint64_t *d_memory;
    checkCudaErrors(cudaMalloc ((void **) &d_memory, sizeof(uint64_t) * M));
    checkCudaErrors(cudaMemcpy((void *)d_memory, memory, sizeof(uint64_t) * M, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    uint64_t *d_offset;
    checkCudaErrors(cudaMalloc ((void **) &d_offset, sizeof(uint64_t) * grid_size));
    uint64_t zero = 0;
    checkCudaErrors(cudaMemset((void *)d_offset, zero, sizeof(uint64_t) * grid_size));


    /* Execute */
    start = getTimeInNSecs();
    access_memory<<<num_wg, num_wi_per_wg>>>(d_memory, d_offset, count);
    cudaThreadSynchronize ();

    cudaError_t error_id = cudaGetLastError();
    if (error_id != cudaSuccess)
    {
        printf("Error is %s\n", cudaGetErrorString(error_id));
    }

    end = getTimeInNSecs();
    kernel_time = end - start;
    printf("GPU Runtime: %.10f %i\n", (double)kernel_time / count, size);
    
    cudaFree(d_memory);
    cudaFree(d_offset);
    delete[] memory;
    delete[] indices;
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[1]));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[0]));

    return 0;
}