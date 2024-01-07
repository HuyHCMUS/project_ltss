#include "conv_utils.h"

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	//TODO
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row + i * m] * B[i+ col*n];
        }
        C[row + col*m] = sum;
    }
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    // Đối với bài này tile_width = blockDim = 32.
    float sum = 0;

    for (int i = 0; i < (n-1)/TILE_WIDTH+1; i++) {
        s_A[threadIdx.y][threadIdx.x] = (Row < m && i*TILE_WIDTH+threadIdx.x < n) ? A[Row + i*TILE_WIDTH+threadIdx.x*m] : 0;

        s_B[threadIdx.y][threadIdx.x] = (Col < k && i*TILE_WIDTH+threadIdx.y < n) ? B[(i*TILE_WIDTH+threadIdx.y)+Col*n] : 0;
        __syncthreads();
    for (int k = 0; k < TILE_WIDTH; ++k)
        sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
    __syncthreads();

    }

    if (Row < m && Col < k)
        C[Row+Col*m] = sum;
}



__global__ void im2col_kernel(const float* image, float* data_col, int height_in, int width_in, 
                            int channel_in, int height_out, int width_out, 
                            int height_kernel, int width_kernel, int stride, int pad_h, int pad_w) 
{
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;

  int step_h = global_idx / width_out;
  int step_w = global_idx % width_out;
  int start_idx = step_h * width_in * stride + step_w * stride;

  for (int c = 0; c < channel_in; c ++) {
    for (int j = 0; j < hw_kernel; j ++) {
      int cur_col = start_idx % width_in + j % width_kernel - pad_w; 
      int cur_row = start_idx / width_in + j / width_kernel - pad_h;

      if (cur_col < 0 || cur_col >= width_in || cur_row < 0 || cur_row >= height_in) {
        //data_col[global_idx * channel_in * hw_kernel +  c * hw_kernel + j] = 0;
        data_col[(c * hw_kernel + j) * hw_out + global_idx] = 0;
      }
      else {
        int pick_idx = cur_row * width_in + cur_col;
        data_col[(c * hw_kernel + j) * hw_out + global_idx] = image[pick_idx + c * height_in * width_in];
      }
    }
  }
}

void matrix_multiplication(float* A, float*B, float*C, int m, int n, int k, const cudaStream_t &stream,int kernel_type){
  float *d_A, *d_B, *d_C;
  CHECK(cudaMalloc(&d_A, m * n * sizeof(float)));
  CHECK(cudaMalloc(&d_B, n * k * sizeof(float)));
  CHECK(cudaMalloc(&d_C,  m * k * sizeof(float)));
  CHECK(cudaMemcpyAsync(d_A, A,  m * n* sizeof(float), cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(d_B, B,  n * k * sizeof(float), cudaMemcpyHostToDevice, stream));

  dim3 blockSize(32, 32);
  dim3 gridSize((k - 1) / blockSize.x + 1, (m - 1) / blockSize.y + 1);
  if (kernel_type == 1) 
    matrix_multiplication_kernel1<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, m, n, k);
  else   if (kernel_type == 1) 
    matrix_multiplication_kernel2<<<gridSize, blockSize, 0, stream>>>(d_A, d_B, d_C, m, n, k);
  CHECK(cudaMemcpyAsync(C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost, stream));
  
  CHECK(cudaFree(d_A));
  CHECK(cudaFree(d_B));
  CHECK(cudaFree(d_C));
}

