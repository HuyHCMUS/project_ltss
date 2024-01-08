/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/conv_gpu1.h"
#include "src/layer/conv_gpu2.h"
#include "src/layer/conv_gpu3.h"

#include "src/layer/fully_connected.h"
#include "src/layer/ave_pooling.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"

void printTiming(std::vector<float> &timings) {
  for (int i = 0; i < timings.size(); i ++) {
    std::cout
      << "L" << i
      << " (" << timings[i] << "s)"
      << (i < timings.size() - 1 ? " -> " : "\n");
  }
}

Layer* make_conv_layer(char* device, int channel_in, int height_in, int width_in, int channel_out, int height_kernel, int width_kernel, int stride = 1, int pad_w = 0, int pad_h = 0) {
  if (strcmp(device, "cpu") == 0) {
    return (Layer *)new Conv(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, stride, pad_w, pad_h);
  }
  if (strcmp(device, "gpu") == 0) {
    return (Layer *)new ConvGPU(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, stride, pad_w, pad_h);
  }
  if (strcmp(device, "gpu_optimize") == 0) {
    return (Layer *)new ConvGPU1(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, stride, pad_w, pad_h);
  }
  if (strcmp(device, "gpu_optimize2") == 0) {
    return (Layer *)new ConvGPU2(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, stride, pad_w, pad_h);
  }
  if (strcmp(device, "gpu_optimize3") == 0) {
    return (Layer *)new ConvGPU3(channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, stride, pad_w, pad_h);
  }
  
  throw std::invalid_argument("unknown device: " + std::string(device));
}

int main(int argc, char* argv[]) {
  // ./demo [run|test] [gpu|cpu]

  if (argc != 3) {
    std::cout << "Usage: ./demo [run|test] [gpu|cpu]" << std::endl;
    return 1;
  }

  char* mode = argv[1];
  std::cout << "Running mode: " << mode << std::endl;

  char* device = argv[2];
  std::cout << "Using device: " << device << std::endl;

  // data
  //MNIST dataset("../data/mnist/");
  MNIST dataset("../data/fashion_mnist/");
  dataset.read();
  int n_train = dataset.train_data.cols();
  int dim_in = dataset.train_data.rows();
  std::cout << "mnist train number: " << n_train << std::endl;
  std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;

  if (strcmp(mode, "test") == 0) {
    int channel_in = 1, height_kernel = 5, width_kernel = 5, channel_out = 6;
    int paramsSize = (channel_in * height_kernel * width_kernel + 1) * channel_out;
    std::vector<float> params(paramsSize);
    for (int i = 0; i < paramsSize; i ++) {
      params[i] = i;
    }

    Layer *base = new Conv(1, 28, 28, channel_out, height_kernel, width_kernel, 1, 2, 2);
    base->set_parameters(params);
    Layer *comp = make_conv_layer(device, 1, 28, 28, channel_out, height_kernel, width_kernel, 1, 2, 2);
    comp->set_parameters(params);

    // Get one sample to test
    Matrix sample = dataset.train_data.col(0);
    std::cout << "Input: " << sample.rows() << "x" << sample.cols() << std::endl;

    std::cout << "Running with base layer..." << std::endl;
    base->forward(sample);

    std::cout << "Running with selected layer..." << std::endl;
    comp->forward(sample);

    std::cout << "Base output: " << base->output().rows() << "x" << base->output().cols() << std::endl;
    std::cout << "Selected output: " << comp->output().rows() << "x" << comp->output().cols() << std::endl;

    Matrix diff = base->output() - comp->output();
    std::cout << "Difference: " << diff.norm() << std::endl;

    // std::cout << comp->output().reshaped(28, 28) << std::endl;
    
    return 0;
  }

  if (strcmp(mode, "run") != 0) {
    std::cout << "Unknown mode: " << mode << std::endl;
    return 1;
  }

  // dnn
  Network dnn;

  // Convolution params
  // channel_in, height_in, width_in, channel_out, height_kernel, width_kernel, 
  // stride = 1, pad_w = 0, pad_h = 0

  // MaxPooling params
  // channel_in, height_in, width_in, height_pool, width_pool, stride = 1


  dnn.add_layer(make_conv_layer(device, 1, 28, 28, 6, 5, 5, 1, 2, 2));
  dnn.add_layer(new ReLU);
  dnn.add_layer(new MaxPooling(6, 28, 28, 2, 2, 2));
  dnn.add_layer(make_conv_layer(device, 6, 14, 14, 16, 5, 5));
  dnn.add_layer(new ReLU);
  Layer* pool = new MaxPooling(16, 10, 10, 2, 2, 2);
  dnn.add_layer(pool);
  dnn.add_layer(new FullyConnected(pool->output_dim(), 120));
  dnn.add_layer(new ReLU);
  dnn.add_layer(new FullyConnected(120, 84));
  dnn.add_layer(new ReLU);
  dnn.add_layer(new FullyConnected(84, 10));
  dnn.add_layer(new Softmax);

  // loss
  Loss* loss = new CrossEntropy;
  dnn.add_loss(loss);
  // train & test
  SGD opt(0.001, 5e-4, 0.9, true);
  // SGD opt(0.001);
  const int n_epoch = 1;
  // const int batch_size = 128;
  const int batch_size = 32;

  for (int epoch = 0; epoch < n_epoch; epoch ++) {
    shuffle_data(dataset.train_data, dataset.train_labels);

    std::vector<float> forward_timings(dnn.num_layers(), 0);

    for (int start_idx = 0; start_idx < n_train; start_idx += batch_size) {
      int ith_batch = start_idx / batch_size;
      Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                    std::min(batch_size, n_train - start_idx));
      Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                    std::min(batch_size, n_train - start_idx));
      Matrix target_batch = one_hot_encode(label_batch, 10);
      if (false && ith_batch % 10 == 1) {
        std::cout << ith_batch << "-th grad: " << std::endl;
        dnn.check_gradient(x_batch, target_batch, 10);
      }
      dnn.forward(x_batch);

      dnn.backward(x_batch, target_batch);
      // display
      if (ith_batch % 50 == 0 && ith_batch > 0) {
        std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss() << std::endl;
        printTiming(forward_timings);
        forward_timings = std::vector<float>(dnn.num_layers(), 0);
      }
      // optimize
      dnn.update(opt);
    }
    // test
    dnn.forward(dataset.test_data);
    float acc = compute_accuracy(dnn.output(), dataset.test_labels);
    std::cout << std::endl;
    std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
    std::cout << std::endl;
  }
  return 0;
}
