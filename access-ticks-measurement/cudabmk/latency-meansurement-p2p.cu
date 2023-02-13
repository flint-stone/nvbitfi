#include <stdio.h> 
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "repeat.h"

typedef unsigned long long int ptrsize_type;

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// add a level of protection to the CUDA SDK samples, let's force samples to
// explicitly include CUDA.H

// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error

// These are the inline versions for all of the SDK helper functions

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

// inline void checkCudaErrors(CUresult err) {
//   if (0 != err) {
//     const char *errorStr = NULL;
//     cuGetErrorString(err, &errorStr);
//     printf(stderr,
//             "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
//             "line %i.\n",
//             err, errorStr, __FILE__, __LINE__);
//   }
// }


__global__ void global_latency (ptrsize_type** my_array, int array_length, int iterations, unsigned long long * duration) {

    unsigned long long int start_time, end_time;
    ptrsize_type *j = (ptrsize_type*)my_array;
    volatile unsigned long long int sum_time;

    sum_time = 0;

    for (int k = 0; k < iterations; k++)
    {

        start_time = clock64();
        repeat1024(j=*(ptrsize_type **)j;)
        end_time = clock64();

        sum_time += (end_time - start_time);
    }

    ((ptrsize_type*)my_array)[array_length] = (ptrsize_type)j;
    ((ptrsize_type*)my_array)[array_length+1] = (ptrsize_type) sum_time;
    duration[0] = sum_time;
}

void parametric_measure_global(int N, int iterations, unsigned long long int maxMem, int stride)
{
    cudaDeviceProp prop[64];
    checkCudaErrors(cudaGetDeviceProperties(&prop[0], 0));
    checkCudaErrors(cudaGetDeviceProperties(&prop[1], 1));
    int p2pCapableGPUs[2]; 
    p2pCapableGPUs[0] = 0;
    p2pCapableGPUs[1] = 1;
    //printf("Enabling peer access...\n");
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[1], 0));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceEnablePeerAccess(p2pCapableGPUs[0], 0));


    unsigned long long int maxMemToArraySize = maxMem / sizeof( ptrsize_type );
    unsigned long long int maxArraySizeNeeded = 1024*iterations*stride;
    unsigned long long int maxArraySize = (maxMemToArraySize<maxArraySizeNeeded)?(maxMemToArraySize):(maxArraySizeNeeded);

    ptrsize_type* h_a = new ptrsize_type[maxArraySize+2];
    ptrsize_type** d_a;
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaMalloc((void **) &d_a, (maxArraySize+2)*sizeof(ptrsize_type)));

    for ( int i = 0; true; i += stride)
    {
        ptrsize_type nextAddr = ((ptrsize_type)d_a)+(i+stride)*sizeof(ptrsize_type);
        if( i+stride < maxArraySize )
        {
            //printf("Initialize entry %i, next addr %lli\n", i, nextAddr);
            h_a[i] = nextAddr;
        }
        else
        {
            h_a[i] = (ptrsize_type)d_a; // point back to the first entry
            break;
        }
    }
    cudaMemcpy((void *)d_a, h_a, (maxArraySize+2)*sizeof(ptrsize_type), cudaMemcpyHostToDevice);

    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    unsigned long long int* duration;
    cudaMalloc ((void **) &duration, sizeof(unsigned long long int));

    unsigned long long int latency_sum = 0;
    int repeat = 1;
    for (int l=0; l <repeat; l++)
    {
        global_latency<<<1,1>>>(d_a, maxArraySize, iterations, duration);
        cudaThreadSynchronize ();

        cudaError_t error_id = cudaGetLastError();
        if (error_id != cudaSuccess)
        {
            printf("Error is %s\n", cudaGetErrorString(error_id));
        }

        unsigned long long int latency;
        cudaMemcpy( &latency, duration, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
        latency_sum += latency;
    }

    cudaFree(d_a);
    cudaFree(duration);

    delete[] h_a;
    //printf("Disabling peer access...\n");
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[0]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[1]));
    checkCudaErrors(cudaSetDevice(p2pCapableGPUs[1]));
    checkCudaErrors(cudaDeviceDisablePeerAccess(p2pCapableGPUs[0]));
    printf("latency average %f\n", (double)(latency_sum/(repeat*1024.0*iterations)) );
}

void measure_global_latency()
{
    int maxMem = 1024*1024*1024; // 1GB
    int N = 1024;
    int iterations = 1;

    for (int stride = 1; stride <= 1024; stride+=1)
    {
        printf (" stride_size  %5d, ", stride*sizeof( ptrsize_type ));
        parametric_measure_global( N, iterations, maxMem, stride );
    }
    for (int stride = 1024; stride <= 1024*1024; stride+=1024)
    {
        printf (" big_stride_size %5d, ", stride*sizeof( ptrsize_type ));
        parametric_measure_global( N, iterations, maxMem, stride );
    }
}

int main()
{
    measure_global_latency();
    return 0;
}