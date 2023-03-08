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
    if(global_id == -1) *p = 1;
}

__global__ void access_memory_std(ulong* memory, ulong* offset, int count, unsigned long long int* d_time, double* avg_arr, double* std_arr){
    unsigned long long int start_time, end_time;
    int global_id = threadIdx.x + blockDim.x * blockIdx.x;
    
    ulong *p = memory;
    p += offset[global_id];
    int global_id_time = global_id * count;
    ulong q;
    ulong sum = 0;
    for(int i=0; i<count; i++){
        start_time = clock(); //clock64();
        q = (ulong)memory + *p;
        p = (ulong*)q; 
        end_time = clock(); //clock64();
        //printf("global_id 1 %d d_time %lli\n", global_id, d_time);
        d_time[global_id_time++] = end_time - start_time;
        //printf("global_id 2 %d %llu\n", global_id_time, end_time - start_time);
        sum += end_time - start_time;
    }
    
    double avg = sum /(double)count;
    double sqr_sum = 0.0;
    global_id_time = global_id * count;
    for(int i = 0; i < count ; i++){
        double diff = d_time[global_id_time + i] - avg;
        //printf("diff %f\n", diff);
        sqr_sum += diff * diff;
    }
    double std = sqrt((double)sqr_sum /count);
    avg_arr[global_id] = avg;
    std_arr[global_id] = std;
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
    bool use_std = false;

    // if (argc != 6) {
    //     printf("./kernel-p2p memory_size num_wg count\n");
    //     exit(-1);
    // }

    size = atoi(argv[1]);
    count = atoi(argv[3]);
    int num_wg = atoi(argv[2]);
    int gpuID = atoi(argv[4]);
    int peerGpuID = atoi(argv[5]);
    int num_wi_per_wg = atoi(argv[5]);

    grid_size = num_wg * num_wi_per_wg;
    workgroup_size = num_wi_per_wg;

    len = size / sizeof(void *);
    int M = pow(2, 32);

    /*P2P*/
    cudaDeviceProp prop[64];
    checkCudaErrors(cudaGetDeviceProperties(&prop[0], gpuID));
    checkCudaErrors(cudaGetDeviceProperties(&prop[1], peerGpuID));
    int p2pCapableGPUs[2]; 
    p2pCapableGPUs[0] = gpuID;
    p2pCapableGPUs[1] = peerGpuID;
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[1], 0));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[0], 0));


    /* Host*/
    uint64_t *memory = new uint64_t[M * sizeof(uint64_t)]; //malloc(M * sizeof(uint64_t));
    uint64_t *indices = new uint64_t[M * sizeof(uint64_t)]; //malloc(M * sizeof(uint64_t));
    uint64_t *offset = new uint64_t[grid_size * sizeof(uint64_t)]; //malloc(M * sizeof(uint64_t));
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

    for(int i = 0; i < grid_size; i++){
        offset[i] = i;
    }

    /* Data copy */
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    uint64_t *d_memory;
    checkCudaErrors(cudaMalloc ((void **) &d_memory, sizeof(uint64_t) * M));
    checkCudaErrors(cudaMemcpy((void *)d_memory, memory, sizeof(uint64_t) * M, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    uint64_t *d_offset;
    checkCudaErrors(cudaMalloc ((void **) &d_offset, sizeof(uint64_t) * grid_size));
    uint64_t zero = 0;
    // checkCudaErrors(cudaMemset((void *)d_offset, zero, sizeof(uint64_t) * grid_size));
    checkCudaErrors(cudaMemcpy((void *)d_offset, offset, sizeof(uint64_t) * grid_size, cudaMemcpyHostToDevice));


    /* Execute */
    if(use_std){
        unsigned long long int *d_time;
        double  *d_avg_arr;
        double *d_std_arr;
        cudaMalloc ((void **) &d_time, sizeof(unsigned long long int) * count * num_wi_per_wg);
        cudaMemset((void *)d_time, (unsigned long long int)1, sizeof(unsigned long long int) * count * num_wi_per_wg);
        cudaMalloc ((void **) &d_avg_arr, sizeof(double) * num_wi_per_wg);
        cudaMemset((void *)d_avg_arr, 0.0, sizeof(double) * num_wi_per_wg);
        cudaMalloc ((void **) &d_std_arr, sizeof(double) * num_wi_per_wg);
        cudaMemset((void *)d_std_arr, 0.0, sizeof(double) * num_wi_per_wg);

        start = getTimeInNSecs();
        access_memory_std<<<num_wg, num_wi_per_wg>>>(d_memory, d_offset, count, d_time, d_avg_arr, d_std_arr);
        cudaThreadSynchronize ();

        cudaError_t error_id = cudaGetLastError();
        if (error_id != cudaSuccess)
        {
            printf("Error is %s\n", cudaGetErrorString(error_id));
        }

        end = getTimeInNSecs();
        double *h_avg_arr = new double[num_wi_per_wg]; 
        double *h_std_arr = new double[num_wi_per_wg]; 
        cudaMemcpy((void *)h_avg_arr, d_avg_arr, sizeof(double) * num_wi_per_wg, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)h_std_arr, d_std_arr, sizeof(double) * num_wi_per_wg, cudaMemcpyDeviceToHost);
        double sum_mean = 0.0;
        double sum_std = 0.0;
        for(int i = 0; i < num_wi_per_wg; i++){
            sum_mean += h_avg_arr[i];
            sum_std += h_std_arr[i];
        }
        printf("GPU Runtime: %.10f %.10f %i\n", (double)sum_mean / num_wi_per_wg, (double) sum_std / num_wi_per_wg, size);
        cudaFree(d_time);
        cudaFree(d_avg_arr);
        cudaFree(d_std_arr);
        delete[] h_avg_arr;
        delete[] h_std_arr;
    }
    else{
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
    }
    
    cudaFree(d_memory);
    cudaFree(d_offset);
    delete[] memory;
    delete[] indices;
    delete[] offset;
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[1]));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[0]));

    return 0;
}