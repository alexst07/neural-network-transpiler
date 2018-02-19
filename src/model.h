#ifndef DROIDNET_MODEL_H
#define DROIDNET_MODEL_H

#include <string>
#include <vector>
#include <cuchar>

#include "schemas/schema_generated.h"

namespace annc {

class FlatBufferModel {
 public:
  FlatBufferModel(const std::string& file);
  ~FlatBufferModel();
  char* data();

 private:
  char *data_;
};

class Buffer {
 public:
  Buffer(std::vector<u_char>&& buf): buf_(std::move(buf)) {}

  const std::vector<u_char>& Data() {
    return buf_;
  }

  const u_char* RawData() {
    return buf_.data();
  }

 private:
  std::vector<u_char> buf_;
};

class Operator {
 public:
  Operator(int index, const std::string& builtin_op_str,
      std::vector<int>&& inputs, std::vector<int>&& outputs)
    : index_(index)
    , builtin_op_str_(builtin_op_str)
    , inputs_(std::move(inputs))
    , outputs_(std::move(outputs)) {}

  int index() const {
    return index_;
  }

  const std::string& builtin_op_str() const {
    return builtin_op_str_;
  }

  const std::vector<int>& inputs() const {
    return inputs_;
  }

  const std::vector<int>& outputs() const {
    return outputs_;
  }

 private:
  int index_;
  std::string builtin_op_str_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
};

class Tensor {
 public:
  Tensor(std::vector<int>&& shape, const std::string& name,
      const Buffer& buffer)
    : shape_(std::move(shape))
    , name_(name)
    , buffer_(buffer) {}

  Tensor(const Tensor& tensor)
    : shape_(tensor.shape_)
    , name_(tensor.name_)
    , buffer_(tensor.buffer_) {}

  Tensor(Tensor&& tensor)
    : shape_(std::move(tensor.shape_))
    , name_(std::move(tensor.name_))
    , buffer_(tensor.buffer_) {}

  const std::string& name() const {
    return name_;
  }

  const std::vector<int>& shape() const {
    return shape_;
  }

 private:
  std::vector<int> shape_;
  std::string name_;
  const Buffer& buffer_;
};

class Graph {
 public:
  Graph() = default;

  void SetInputs(std::vector<int>&& inputs) {
    inputs_ = std::move(inputs);
  }

  void SetOutputs(std::vector<int>&& outputs) {
    outputs_ = std::move(outputs);
  }

  void AddTensor(const Tensor& tensor) {
    tensors_.push_back(tensor);
  }

  void AddOperator(const Operator& op) {
    operators_.push_back(op);
  }

  const std::vector<Tensor>& Tensors() const {
    return tensors_;
  }

  const std::vector<Operator>& Operators() const {
    return operators_;
  }

  const std::vector<int>& Inputs() const {
    return inputs_;
  }

  const std::vector<int>& Outputs() const {
    return outputs_;
  }

 private:
  std::vector<Tensor> tensors_;
  std::vector<Operator> operators_;
  std::vector<int> inputs_;
  std::vector<int> outputs_;
};

class Model {
 public:
  Model(const std::string& file);

  const char* description();

  Graph& graph() {
    return graph_;
  }

 private:
  void PopulateGraph();

  void PopulateGraphInputs(const tflite::SubGraph* graph);

  void PopulateGraphOutputs(const tflite::SubGraph* graph);

  void PopulateGraphTensors(const tflite::SubGraph* graph);

  void PopulateGraphOperators(const tflite::SubGraph* graph);

  void PopulateBuffers();

  FlatBufferModel flat_buffers_;
  const tflite::Model *fb_model_;
  std::vector<Buffer> buffers_;
  Graph graph_;
};

}

#endif
