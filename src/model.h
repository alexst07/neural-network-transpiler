#ifndef DROIDNET_MODEL_H
#define DROIDNET_MODEL_H

#include <string>
#include <vector>
#include <memory>
#include <cuchar>
#include <boost/variant.hpp>

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

enum class ActivationFunctionType: int8_t {
  NONE,
  RELU,
  RELU1,
  RELU6,
  TANH,
  SIGN_BIT
};

enum class Padding: int8_t { SAME, VALID };

enum class BuiltinOptionsType {
  Conv2DOptions,
  DepthwiseConv2DOptions,
  ConcatEmbeddingsOptions,
  LSHProjectionOptions,
  Pool2DOptions,
  SVDFOptions,
  RNNOptions,
  FullyConnectedOptions,
  SoftmaxOptions,
  ConcatenationOptions,
  AddOptions,
  L2NormOptions,
  LocalResponseNormalizationOptions,
  LSTMOptions,
  ResizeBilinearOptions,
  CallOptions,
  ReshapeOptions,
  SkipGramOptions,
  SpaceToDepthOptions,
  EmbeddingLookupSparseOptions,
  MulOptions,
  PadOptions,
  GatherOptions,
  BatchToSpaceNDOptions,
  SpaceToBatchNDOptions,
  TransposeOptions,
  MeanOptions,
  SubOptions,
  DivOptions,
  SqueezeOptions,
  SequenceRNNOptions
};

struct BuiltinOptions {
  BuiltinOptions(BuiltinOptionsType type_op): type(type_op) {}

  BuiltinOptionsType type;
};

struct Conv2DOptions: public BuiltinOptions {
  Conv2DOptions(): BuiltinOptions(BuiltinOptionsType::Conv2DOptions) {}

  Padding padding;
  int stride_w;
  int stride_h;
  ActivationFunctionType fused_activation_function;
};

struct Pool2DOptions: public BuiltinOptions {
  Pool2DOptions(): BuiltinOptions(BuiltinOptionsType::Pool2DOptions) {}

  Padding padding;
  int stride_w;
  int stride_h;
  int filter_width;
  int filter_height;
  ActivationFunctionType fused_activation_function;
};

struct DepthwiseConv2DOptions: public BuiltinOptions {
  DepthwiseConv2DOptions()
    : BuiltinOptions(BuiltinOptionsType::DepthwiseConv2DOptions) {}

  Padding padding;
  int stride_w;
  int stride_h;
  int depth_multiplier;
  ActivationFunctionType fused_activation_function;
};

struct ConcatEmbeddingsOptions: public BuiltinOptions {
  ConcatEmbeddingsOptions()
    : BuiltinOptions(BuiltinOptionsType::ConcatEmbeddingsOptions) {}

  int num_channels;
  std::vector<int> num_columns_per_channel;
  std::vector<int> embedding_dim_per_channel;
};

enum class LSHProjectionType: int8_t {
  UNKNOWN = 0,
  SPARSE = 1,
  DENSE = 2,
};

struct LSHProjectionOptions: public BuiltinOptions {
  LSHProjectionOptions()
    : BuiltinOptions(BuiltinOptionsType::LSHProjectionOptions) {}

  LSHProjectionType type;
};

struct SVDFOptions: public BuiltinOptions {
  SVDFOptions(): BuiltinOptions(BuiltinOptionsType::SVDFOptions) {}

  int rank;
  ActivationFunctionType fused_activation_function;
};

struct RNNOptions: public BuiltinOptions {
  RNNOptions(): BuiltinOptions(BuiltinOptionsType::RNNOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct SequenceRNNOptions: public BuiltinOptions {
  SequenceRNNOptions()
    : BuiltinOptions(BuiltinOptionsType::SequenceRNNOptions) {}

  bool time_major;
  ActivationFunctionType fused_activation_function;
};

struct FullyConnectedOptions: public BuiltinOptions {
  FullyConnectedOptions()
    : BuiltinOptions(BuiltinOptionsType::FullyConnectedOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct SoftmaxOptions: public BuiltinOptions {
  SoftmaxOptions()
    : BuiltinOptions(BuiltinOptionsType::SoftmaxOptions) {}

  float beta;
};

struct ConcatenationOptions: public BuiltinOptions {
  ConcatenationOptions()
    : BuiltinOptions(BuiltinOptionsType::ConcatenationOptions) {}

  int axis;
  ActivationFunctionType fused_activation_function;
};

struct AddOptions: public BuiltinOptions {
  AddOptions(): BuiltinOptions(BuiltinOptionsType::AddOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct MulOptions: public BuiltinOptions {
  MulOptions(): BuiltinOptions(BuiltinOptionsType::MulOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct L2NormOptions: public BuiltinOptions {
  L2NormOptions(): BuiltinOptions(BuiltinOptionsType::L2NormOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct LocalResponseNormalizationOptions: public BuiltinOptions {
  LocalResponseNormalizationOptions()
    : BuiltinOptions(BuiltinOptionsType::LocalResponseNormalizationOptions) {}

  int radius;
  float bias;
  float alpha;
  float beta;
};

struct LSTMOptions: public BuiltinOptions {
  LSTMOptions(): BuiltinOptions(BuiltinOptionsType::LSTMOptions) {}

  float cell_clip;
  float proj_clip;
  ActivationFunctionType fused_activation_function;
};

struct ResizeBilinearOptions: public BuiltinOptions {
  ResizeBilinearOptions()
    : BuiltinOptions(BuiltinOptionsType::ResizeBilinearOptions) {}

  int new_height;
  int new_width;
};

struct CallOptions: public BuiltinOptions {
  CallOptions(): BuiltinOptions(BuiltinOptionsType::CallOptions) {}

  uint subgraph;
};

struct PadOptions: public BuiltinOptions {
  PadOptions(): BuiltinOptions(BuiltinOptionsType::PadOptions) {}

  std::vector<int> before_padding;
  std::vector<int> after_padding;
};

struct ReshapeOptions: public BuiltinOptions {
  ReshapeOptions(): BuiltinOptions(BuiltinOptionsType::ReshapeOptions) {}

  std::vector<int> new_shape;
};

struct SpaceToBatchNDOptions: public BuiltinOptions {
  SpaceToBatchNDOptions()
    : BuiltinOptions(BuiltinOptionsType::SpaceToBatchNDOptions) {}

  std::vector<int> block_shape;
  std::vector<int> before_paddings;
  std::vector<int> after_paddings;
};

struct BatchToSpaceNDOptions: public BuiltinOptions {
  BatchToSpaceNDOptions()
    : BuiltinOptions(BuiltinOptionsType::BatchToSpaceNDOptions) {}

  std::vector<int> block_shape;
  std::vector<int> before_crops;
  std::vector<int> after_crops;
};

struct SkipGramOptions: public BuiltinOptions {
  SkipGramOptions(): BuiltinOptions(BuiltinOptionsType::SkipGramOptions) {}

  int ngram_size;
  int max_skip_size;
  bool include_all_ngrams;
};

struct SpaceToDepthOptions: public BuiltinOptions {
  SpaceToDepthOptions()
    : BuiltinOptions(BuiltinOptionsType::SpaceToDepthOptions) {}

  int block_size;
};

struct SubOptions: public BuiltinOptions {
  SubOptions(): BuiltinOptions(BuiltinOptionsType::SubOptions) {}

  ActivationFunctionType fused_activation_function;
};

struct DivOptions: public BuiltinOptions {
  DivOptions(): BuiltinOptions(BuiltinOptionsType::DivOptions) {}

  ActivationFunctionType fused_activation_function;
};

enum class CombinerType : int8_t {
  SUM = 0,
  MEAN = 1,
  SQRTN = 2,
};

struct EmbeddingLookupSparseOptions: public BuiltinOptions {
  EmbeddingLookupSparseOptions()
    : BuiltinOptions(BuiltinOptionsType::EmbeddingLookupSparseOptions) {}

  CombinerType combiner;
};

struct GatherOptions: public BuiltinOptions {
  GatherOptions(): BuiltinOptions(BuiltinOptionsType::GatherOptions) {}

  int axis;
};

struct TransposeOptions: public BuiltinOptions {
  TransposeOptions(): BuiltinOptions(BuiltinOptionsType::TransposeOptions) {}

  std::vector<int> perm;
};

struct MeanOptions: public BuiltinOptions {
  MeanOptions(): BuiltinOptions(BuiltinOptionsType::MeanOptions) {}

  std::vector<int> axis;
  bool keep_dims;
};

struct SqueezeOptions: public BuiltinOptions {
  SqueezeOptions(): BuiltinOptions(BuiltinOptionsType::SqueezeOptions) {}

  std::vector<int> squeeze_dims;
};

class Operator {
 public:
  Operator(int index, const std::string& builtin_op_str,
      std::vector<int>&& inputs, std::vector<int>&& outputs)
    : index_(index)
    , builtin_op_str_(builtin_op_str)
    , inputs_(std::move(inputs))
    , outputs_(std::move(outputs)) {}

  Operator(Operator&& op)
    : index_(op.index_)
    , builtin_op_str_(std::move(op.builtin_op_str_))
    , inputs_(std::move(op.inputs_))
    , outputs_(std::move(op.outputs_))
    , builtin_op_(std::move(op.builtin_op_)) {}

  Operator(const Operator&) = delete;

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
  std::unique_ptr<BuiltinOptions> builtin_op_;
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

  void AddTensor(Tensor&& tensor) {
    tensors_.push_back(std::move(tensor));
  }

  void AddOperator(Operator&& op) {
    operators_.push_back(std::move(op));
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

  std::unique_ptr<BuiltinOptions> HandleBuiltinOptions(
      const tflite::Operator* op);

  ActivationFunctionType ConvertActivationFunction(
      tflite::ActivationFunctionType fn_activation_type);

  FlatBufferModel flat_buffers_;
  const tflite::Model *fb_model_;
  std::vector<Buffer> buffers_;
  Graph graph_;
};

}

#endif
