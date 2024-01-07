#ifndef __CONV_UTILS__
#define __CONV_UTILS__

#include <Eigen/Core>
#include <cuda_runtime_api.h>
#define TILE_WIDTH 32


#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

#define ceilDiv(x, y) ((x - 1) / y + 1)

struct GpuTimer
{
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer()
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start()
	{
		cudaEventRecord(start, 0);                                                                 
		cudaEventSynchronize(start);
	}

	void Stop()
	{
		cudaEventRecord(stop, 0);
	}

	float Elapsed()
	{
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k);
__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k);
//void matrix_multiplication(float* A, float*B, float*C, int m, int n, int k, 
//                            const cudaStream_t &stream);
__global__ void im2col_kernel(const float* image, float* data_col, int height_in, int width_in, 
                            int channel_in, int height_out, int width_out, 
                            int height_kernel, int width_kernel, int stride, int pad_h, int pad_w); 
void matrix_multiplication(float* A, float*B, float*C, int m, int n, int k, const cudaStream_t &stream,int kernel_type);


#endif