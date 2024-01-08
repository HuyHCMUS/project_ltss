#include "conv_gpu.h"
#include <math.h>
#include <iostream>

void ConvGPU::init() {
  height_out = (1 + (height_in - height_kernel + 2 * pad_h) / stride);
  width_out =   (1 + (width_in - width_kernel + 2 * pad_w) / stride);
  dim_out = height_out * width_out * channel_out;

  weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  bias.resize(channel_out);
  grad_weight.resize(channel_in * height_kernel * width_kernel, channel_out);
  grad_bias.resize(channel_out);
  set_normal_random(weight.data(), weight.size(), 0, 0.01);
  set_normal_random(bias.data(), bias.size(), 0, 0.01);
  //std::cout << weight.colwise().sum() << std::endl;
  //std::cout << weight.colwise().sum() + bias.transpose() << std::endl;
}

// im2col, used for bottom
// image size: Vector (height_in * width_in * channel_in)
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
void ConvGPU::im2col(const Vector& image, Matrix& data_col) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // im2col
  data_col.resize(hw_out, hw_kernel * channel_in);
  for (int c = 0; c < channel_in; c ++) {
    Vector map = image.block(hw_in * c, 0, hw_in, 1);  // c-th channel map
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          data_col(i, c * hw_kernel + j) = 0;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          data_col(i, c * hw_kernel + j) = map(pick_idx);  // pick which pixel
        }
      }
    }
  }
}

void ConvGPU::forward(const Matrix& btm) {
  Matrix bottom = btm;

  int n_sample = bottom.cols();
  top.resize(height_out * width_out * channel_out, n_sample);
  data_cols.resize(n_sample);

  int hw_in = height_in * width_in;
  int hw_out = height_out * width_out;
  int hw_kernel = height_kernel * width_kernel;

  top.setZero();
  top.transposeInPlace();
  bottom.transposeInPlace();

  // std::cout << "bottom: " << bottom.rows() << " " << bottom.cols() << std::endl
  //           << "weight: " << weight.rows() << " " << weight.cols() << std::endl
  //           << "bias: " << bias.rows() << " " << bias.cols() << std::endl
  //           << "top: " << top.rows() << " " << top.cols() << std::endl;

  float *d_bottom, *d_weight, *d_top;

  CHECK(cudaMalloc(&d_bottom, bottom.rows() * bottom.cols() * sizeof(float)));
  CHECK(cudaMalloc(&d_weight, weight.rows() * weight.cols() * sizeof(float)));
  CHECK(cudaMalloc(&d_top, top.rows() * top.cols() * sizeof(float)));

  CHECK(cudaMemcpy(d_bottom, bottom.data(), bottom.rows() * bottom.cols() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_weight, weight.data(), weight.rows() * weight.cols() * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_top, top.data(), top.rows() * top.cols() * sizeof(float), cudaMemcpyHostToDevice));

  for (int i = 0; i < n_sample; i ++) {
    // im2col
    Matrix data_col;
    im2col(bottom.row(i), data_col);
    data_cols[i] = data_col;

    // bottom:  (1x28x28, N)    (6x14x14, N)
    // weight:  (5x5, 6)        (6x5x5, 16)
    // bias:    (6, 1)          (16, 1)
    // top:     (6x28x28, N)    (16x10x10, N)

    // now
    // bottom:  (N, 1x28x28)    (N, 6x14x14)
    // weight:  (5x5, 6)        (6x5x5, 16)
    // bias:    (6, 1)          (16, 1)
    // top:     (N, 6x28x28)    (N, 16x10x10)

    // top = bottom * weight + bias

    for (int c_in = 0; c_in < channel_in; c_in++) {
      for (int c_out = 0; c_out < channel_out; c_out++) {
        dim3 blockSize(32, 32);
        dim3 gridSize(ceilDiv(width_out, blockSize.x), ceilDiv(height_out, blockSize.y));

        kernelConvolution<<<gridSize, blockSize>>>(
          &d_bottom[i * bottom.cols() + c_in * hw_in], width_in, height_in,
          &d_weight[c_out * hw_kernel], width_kernel, height_kernel,
          &d_top[i * bottom.cols() + c_out * hw_out], width_out, height_out,
          pad_w, pad_h
        );
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
      }
    }
  }

  CHECK(cudaMemcpy(top.data(), d_top, top.rows() * top.cols() * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(d_top));
  CHECK(cudaFree(d_bottom));
  CHECK(cudaFree(d_weight));

  for (int c = 0; c < channel_out; c ++) {
    top.middleCols(hw_out * c, hw_out).array() += bias(c, 0);
  }
  // for (int i = 0; i < n_sample; i ++) {
  // }

  bottom.transposeInPlace();
  top.transposeInPlace();
}

__global__ void kernelConvolution(
  float* inp, int h_in, int w_in,
  float* weight, int h_kernel, int w_kernel,
  float* out, int h_out, int w_out,
  int pad_w = 0, int pad_h = 0
) {
  int r_out = blockIdx.y * blockDim.y + threadIdx.y;
  int c_out = blockIdx.x * blockDim.x + threadIdx.x;

  if (r_out >= h_out || c_out >= w_out) {
    return;
  }

  float sum = 0;

  for (int r_off = - h_kernel / 2; r_off <= h_kernel / 2; r_off++) {
    for (int c_off = - w_kernel / 2; c_off <= w_kernel / 2; c_off++) {
      int r_kernel = r_off + h_kernel / 2;
      int c_kernel = c_off + w_kernel / 2;
      float weight_val = weight[r_kernel * w_kernel + c_kernel];

      int w_diff = w_in + pad_w * 2 - w_out;
      int h_diff = h_in + pad_h * 2 - h_out;

      int r_in = r_out + h_diff / 2 - h_kernel / 2 + r_off;
      int c_in = c_out + w_diff / 2 - w_kernel / 2 + c_off;
      float inp_val = (
        0 <= r_in && r_in < h_in &&
        0 <= c_in && c_in < w_in
        ? inp[r_in * w_in + c_in]
        : 0
      );

      sum += inp_val * weight_val;
    }
  }

  atomicAdd(&out[r_out * w_out + c_out], sum);
}

// col2im, used for grad_bottom
// data_col size: Matrix (hw_out, hw_kernel * channel_in)
// image size: Vector (height_in * width_in * channel_in)
void ConvGPU::col2im(const Matrix& data_col, Vector& image) {
  int hw_in = height_in * width_in;
  int hw_kernel = height_kernel * width_kernel;
  int hw_out = height_out * width_out;
  // col2im
  image.resize(hw_in * channel_in);
  image.setZero();
  for (int c = 0; c < channel_in; c ++) {
    for (int i = 0; i < hw_out; i ++) {
      int step_h = i / width_out;
      int step_w = i % width_out;
      int start_idx = step_h * width_in * stride + step_w * stride;  // left-top idx of window
      for (int j = 0; j < hw_kernel; j ++) {
        int cur_col = start_idx % width_in + j % width_kernel - pad_w;  // col after padding
        int cur_row = start_idx / width_in + j / width_kernel - pad_h;
        if (cur_col < 0 || cur_col >= width_in || cur_row < 0 ||
            cur_row >= height_in) {
          continue;
        }
        else {
          //int pick_idx = start_idx + (j / width_kernel) * width_in + j % width_kernel;
          int pick_idx = cur_row * width_in + cur_col;
          image(c * hw_in + pick_idx) += data_col(i, c * hw_kernel + j);  // pick which pixel
        }
      }
    }
  }
}

void ConvGPU::backward(const Matrix& bottom, const Matrix& grad_top) {
  int n_sample = bottom.cols();
  grad_weight.setZero();
  grad_bias.setZero();
  grad_bottom.resize(height_in * width_in * channel_in, n_sample);
  grad_bottom.setZero();
  for (int i = 0; i < n_sample; i ++) {
    // im2col of grad_top
    Matrix grad_top_i = grad_top.col(i);
    Matrix grad_top_i_col = Eigen::Map<Matrix>(grad_top_i.data(),
                              height_out * width_out, channel_out);
    // d(L)/d(w) = \sum{ d(L)/d(z_i) * d(z_i)/d(w) }
    grad_weight += data_cols[i].transpose() * grad_top_i_col;
    // d(L)/d(b) = \sum{ d(L)/d(z_i) * d(z_i)/d(b) }
    grad_bias += grad_top_i_col.colwise().sum().transpose();
    // d(L)/d(x) = \sum{ d(L)/d(z_i) * d(z_i)/d(x) } = d(L)/d(z)_col * w'
    Matrix grad_bottom_i_col = grad_top_i_col * weight.transpose();
    // col2im of grad_bottom
    Vector grad_bottom_i;
    col2im(grad_bottom_i_col, grad_bottom_i);
    grad_bottom.col(i) = grad_bottom_i;
  }
}

void ConvGPU::update(Optimizer& opt) {
  Vector::AlignedMapType weight_vec(weight.data(), weight.size());
  Vector::AlignedMapType bias_vec(bias.data(), bias.size());
  Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
  Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

  opt.update(weight_vec, grad_weight_vec);
  opt.update(bias_vec, grad_bias_vec);
}

std::vector<float> ConvGPU::get_parameters() const {
  std::vector<float> res(weight.size() + bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(weight.data(), weight.data() + weight.size(), res.begin());
  std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
  return res;
}

void ConvGPU::set_parameters(const std::vector<float>& param) {
  if(static_cast<int>(param.size()) != weight.size() + bias.size())
      throw std::invalid_argument("Parameter size does not match");
  std::copy(param.begin(), param.begin() + weight.size(), weight.data());
  std::copy(param.begin() + weight.size(), param.end(), bias.data());
}

std::vector<float> ConvGPU::get_derivatives() const {
  std::vector<float> res(grad_weight.size() + grad_bias.size());
  // Copy the data of weights and bias to a long vector
  std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
  std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(),
            res.begin() + grad_weight.size());
  return res;
}
