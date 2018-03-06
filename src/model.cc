#include "model.h"

#include <cstdio>
#include <iostream>

#include "exception.h"

namespace annc {

FlatBufferModel::FlatBufferModel(const std::string& fname) {
  FILE* file = fopen(fname.c_str(), "rb");
  fseek(file, 0L, SEEK_END);
  int length = ftell(file);
  fseek(file, 0L, SEEK_SET);
  data_ = new char[length];
  fread(data_, sizeof(char), length, file);
  fclose(file);
}

FlatBufferModel::~FlatBufferModel() {
  delete data_;
}

char* FlatBufferModel::data() {
  return data_;
}

Model::Model(const std::string& fname)
  : flat_buffers_(fname)
  , fb_model_(tflite::GetModel(flat_buffers_.data())) {
  PopulateBuffers();
  PopulateGraph();
}

void Model::PopulateGraphInputs(const tflite::SubGraph* graph) {
  std::vector<int> inputs = AssignVector<int>(graph->inputs());
  graph_.SetInputs(std::move(inputs));
}

void Model::PopulateGraphOutputs(const tflite::SubGraph* graph) {
  std::vector<int> outputs = AssignVector<int>(graph->outputs());
  graph_.SetOutputs(std::move(outputs));
}

TensorType Model::ConvertTensorType(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType::FLOAT32:
      return TensorType::FLOAT32;
      break;

    case tflite::TensorType::FLOAT16:
      return TensorType::FLOAT16;
      break;

    case tflite::TensorType::INT32:
      return TensorType::INT32;
      break;

    case tflite::TensorType::UINT8:
      return TensorType::UINT8;
      break;

    case tflite::TensorType::INT64:
      return TensorType::INT64;
      break;

    case tflite::TensorType::STRING:
      return TensorType::STRING;
      break;

    default:
      FATAL("Tensor type not valid")
  }
}

void Model::PopulateGraphTensors(const tflite::SubGraph* graph) {
  auto tensors = graph->tensors();

  // get tensors
  for (auto it = tensors->begin(); it != tensors->end(); ++it) {
    std::vector<int> vec_shape = AssignVector<int>(it->shape());
    std::string name = it->name()->c_str();
    uint buf_index = it->buffer();
    const Buffer& buffer = buffers_[buf_index];

    // get quantization
    auto quantization = it->quantization();
    std::unique_ptr<QuantizationParameters> quantization_ptr(
        new QuantizationParameters);

    if (quantization) {
      quantization_ptr->min = AssignVector<float>(quantization->min());
      quantization_ptr->max = AssignVector<float>(quantization->max());
      quantization_ptr->scale = AssignVector<float>(quantization->scale());
      quantization_ptr->zero_point =
          AssignVector<long>(quantization->zero_point());
    }

    TensorType type = ConvertTensorType(it->type());
    graph_.AddTensor(std::move(Tensor(std::move(vec_shape), type, name, buffer,
        buf_index, std::move(quantization_ptr))));
  }
}

ActivationFunctionType Model::ConvertActivationFunction(
    tflite::ActivationFunctionType fn_activation_type) {
  switch (fn_activation_type) {
    case tflite::ActivationFunctionType::NONE:
      return ActivationFunctionType::NONE;
      break;

    case tflite::ActivationFunctionType::RELU:
      return ActivationFunctionType::RELU;
      break;

    case tflite::ActivationFunctionType::RELU_N1_TO_1:
      return ActivationFunctionType::NONE;
      break;

    case tflite::ActivationFunctionType::RELU6:
      return ActivationFunctionType::RELU6;
      break;

    case tflite::ActivationFunctionType::TANH:
      return ActivationFunctionType::TANH;
      break;

    case tflite::ActivationFunctionType::SIGN_BIT:
      return ActivationFunctionType::SIGN_BIT;
      break;

    default:
      return ActivationFunctionType::NONE;
  }
}

Padding Model::ConvertPadding(tflite::Padding padding) {
  if (padding == tflite::Padding::SAME) {
    return Padding::SAME;
  } else {
    return Padding::VALID;
  }
}

std::unique_ptr<NoneOptions> Model::MakeNoneOptions(
    const tflite::Operator* /*op*/) {
  std::unique_ptr<NoneOptions> option = std::make_unique<NoneOptions>();
  return option;
}

std::unique_ptr<Conv2DOptions> Model::MakeConv2DOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::Conv2DOptions*>(
      op->builtin_options());

  std::unique_ptr<Conv2DOptions> option = std::make_unique<Conv2DOptions>();

  option->stride_w = p->stride_w();
  option->stride_h = p->stride_h();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());
  option->padding = ConvertPadding(p->padding());

  return option;
}

std::unique_ptr<Pool2DOptions> Model::MakePool2DOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::Pool2DOptions*>(
      op->builtin_options());

  std::unique_ptr<Pool2DOptions> option = std::make_unique<Pool2DOptions>();

  option->stride_w = p->stride_w();
  option->stride_h = p->stride_h();
  option->filter_width = p->filter_width();
  option->filter_height = p->filter_height();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());
  option->padding = ConvertPadding(p->padding());

  return option;
}

std::unique_ptr<DepthwiseConv2DOptions> Model::MakeDepthwiseConv2DOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::DepthwiseConv2DOptions*>(
      op->builtin_options());

  std::unique_ptr<DepthwiseConv2DOptions> option =
      std::make_unique<DepthwiseConv2DOptions>();

  option->stride_w = p->stride_w();
  option->stride_h = p->stride_h();
  option->depth_multiplier = p->depth_multiplier();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());
  option->padding = ConvertPadding(p->padding());

  return option;
}

std::unique_ptr<ConcatEmbeddingsOptions> Model::MakeConcatEmbeddingsOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::ConcatEmbeddingsOptions*>(
      op->builtin_options());

  std::unique_ptr<ConcatEmbeddingsOptions> option =
      std::make_unique<ConcatEmbeddingsOptions>();

  option->num_channels = p->num_channels();

  auto num_columns = p->num_columns_per_channel();
  option->num_columns_per_channel = AssignVector<int>(num_columns);

  auto embedding_dim = p->embedding_dim_per_channel();
  option->embedding_dim_per_channel = AssignVector<int>(embedding_dim);

  return option;
}

std::unique_ptr<LSHProjectionOptions> Model::MakeLSHProjectionOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::LSHProjectionOptions*>(
      op->builtin_options());

  std::unique_ptr<LSHProjectionOptions> option =
      std::make_unique<LSHProjectionOptions>();

  switch (p->type()) {
    case tflite::LSHProjectionType::UNKNOWN:
      option->type = LSHProjectionType::UNKNOWN;
      break;

    case tflite::LSHProjectionType::SPARSE:
      option->type = LSHProjectionType::SPARSE;
      break;

    case tflite::LSHProjectionType::DENSE:
      option->type = LSHProjectionType::DENSE;
      break;
  }

  return option;
}

std::unique_ptr<SVDFOptions> Model::MakeSVDFOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SVDFOptions*>(op->builtin_options());

  std::unique_ptr<SVDFOptions> option = std::make_unique<SVDFOptions>();

  option->rank = p->rank();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<RNNOptions> Model::MakeRNNOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::RNNOptions*>(op->builtin_options());

  std::unique_ptr<RNNOptions> option = std::make_unique<RNNOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<SequenceRNNOptions> Model::MakeSequenceRNNOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SequenceRNNOptions*>(
      op->builtin_options());

  std::unique_ptr<SequenceRNNOptions> option =
      std::make_unique<SequenceRNNOptions>();

  option->time_major = p->time_major();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<FullyConnectedOptions> Model::MakeFullyConnectedOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::FullyConnectedOptions*>(
      op->builtin_options());

  std::unique_ptr<FullyConnectedOptions> option =
      std::make_unique<FullyConnectedOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<SoftmaxOptions> Model::MakeSoftmaxOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SoftmaxOptions*>(
      op->builtin_options());

  std::unique_ptr<SoftmaxOptions> option = std::make_unique<SoftmaxOptions>();

  option->beta = p->beta();

  return option;
}

std::unique_ptr<ConcatenationOptions> Model::MakeConcatenationOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::ConcatenationOptions*>(
      op->builtin_options());

  std::unique_ptr<ConcatenationOptions> option =
      std::make_unique<ConcatenationOptions>();

  option->axis = p->axis();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<AddOptions> Model::MakeAddOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::AddOptions*>(op->builtin_options());

  std::unique_ptr<AddOptions> option = std::make_unique<AddOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<MulOptions> Model::MakeMulOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::MulOptions*>(op->builtin_options());

  std::unique_ptr<MulOptions> option = std::make_unique<MulOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<L2NormOptions> Model::MakeL2NormOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::L2NormOptions*>(
      op->builtin_options());

  std::unique_ptr<L2NormOptions> option = std::make_unique<L2NormOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<LocalResponseNormalizationOptions>
Model::MakeLocalResponseNormalizationOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::LocalResponseNormalizationOptions*>(
      op->builtin_options());

  std::unique_ptr<LocalResponseNormalizationOptions> option =
      std::make_unique<LocalResponseNormalizationOptions>();

  option->radius = p->radius();
  option->bias = p->bias();
  option->alpha = p->alpha();
  option->beta = p->beta();

  return option;
}

std::unique_ptr<LSTMOptions> Model::MakeLSTMOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::LSTMOptions*>(op->builtin_options());

  std::unique_ptr<LSTMOptions> option = std::make_unique<LSTMOptions>();

  option->cell_clip = p->cell_clip();
  option->proj_clip = p->proj_clip();
  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<ResizeBilinearOptions> Model::MakeResizeBilinearOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::ResizeBilinearOptions*>(
      op->builtin_options());

  std::unique_ptr<ResizeBilinearOptions> option =
      std::make_unique<ResizeBilinearOptions>();

  option->new_height = p->new_height();
  option->new_width = p->new_width();

  return option;
}

std::unique_ptr<CallOptions> Model::MakeCallOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::CallOptions*>(op->builtin_options());

  std::unique_ptr<CallOptions> option = std::make_unique<CallOptions>();

  option->subgraph = p->subgraph();

  return option;
}

std::unique_ptr<PadOptions> Model::MakePadOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::PadOptions*>(op->builtin_options());

  std::unique_ptr<PadOptions> option = std::make_unique<PadOptions>();
  option->before_padding = AssignVector<int>(p->before_padding());
  option->after_padding = AssignVector<int>(p->after_padding());

  return option;
}

std::unique_ptr<ReshapeOptions> Model::MakeReshapeOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::ReshapeOptions*>(
      op->builtin_options());

  std::unique_ptr<ReshapeOptions> option = std::make_unique<ReshapeOptions>();

  option->new_shape = AssignVector<int>(p->new_shape());

  return option;
}

std::unique_ptr<SpaceToBatchNDOptions> Model::MakeSpaceToBatchNDOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SpaceToBatchNDOptions*>(
      op->builtin_options());

  std::unique_ptr<SpaceToBatchNDOptions> option =
      std::make_unique<SpaceToBatchNDOptions>();

  option->block_shape = AssignVector<int>(p->block_shape());
  option->before_paddings = AssignVector<int>(p->before_paddings());
  option->after_paddings = AssignVector<int>(p->after_paddings());

  return option;
}

std::unique_ptr<BatchToSpaceNDOptions> Model::MakeBatchToSpaceNDOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::BatchToSpaceNDOptions*>(
      op->builtin_options());

  std::unique_ptr<BatchToSpaceNDOptions> option =
      std::make_unique<BatchToSpaceNDOptions>();

  option->block_shape = AssignVector<int>(p->block_shape());
  option->before_crops = AssignVector<int>(p->before_crops());
  option->after_crops = AssignVector<int>(p->after_crops());

  return option;
}

std::unique_ptr<SkipGramOptions> Model::MakeSkipGramOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SkipGramOptions*>(
      op->builtin_options());

  std::unique_ptr<SkipGramOptions> option = std::make_unique<SkipGramOptions>();

  option->ngram_size = p->ngram_size();
  option->max_skip_size = p->max_skip_size();
  option->include_all_ngrams = p->include_all_ngrams();

  return option;
}

std::unique_ptr<SpaceToDepthOptions> Model::MakeSpaceToDepthOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SpaceToDepthOptions*>(
      op->builtin_options());

  std::unique_ptr<SpaceToDepthOptions> option =
      std::make_unique<SpaceToDepthOptions>();

  option->block_size = p->block_size();

  return option;
}

std::unique_ptr<SubOptions> Model::MakeSubOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SubOptions*>(op->builtin_options());

  std::unique_ptr<SubOptions> option = std::make_unique<SubOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<DivOptions> Model::MakeDivOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::DivOptions*>(op->builtin_options());

  std::unique_ptr<DivOptions> option = std::make_unique<DivOptions>();

  option->fused_activation_function = ConvertActivationFunction(
      p->fused_activation_function());

  return option;
}

std::unique_ptr<EmbeddingLookupSparseOptions>
Model::MakeEmbeddingLookupSparseOptions(const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::EmbeddingLookupSparseOptions*>(
      op->builtin_options());

  std::unique_ptr<EmbeddingLookupSparseOptions> option =
      std::make_unique<EmbeddingLookupSparseOptions>();

  switch (p->combiner()) {
    case tflite::CombinerType::SUM:
      option->combiner = CombinerType::SUM;
      break;

    case tflite::CombinerType::MEAN:
      option->combiner = CombinerType::MEAN;
      break;

    case tflite::CombinerType::SQRTN:
      option->combiner = CombinerType::SQRTN;
      break;
  }

  return option;
}

std::unique_ptr<GatherOptions> Model::MakeGatherOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::GatherOptions*>(
      op->builtin_options());

  std::unique_ptr<GatherOptions> option = std::make_unique<GatherOptions>();

  option->axis = p->axis();

  return option;
}

std::unique_ptr<TransposeOptions> Model::MakeTransposeOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::TransposeOptions*>(
      op->builtin_options());

  std::unique_ptr<TransposeOptions> option =
      std::make_unique<TransposeOptions>();

  option->perm = AssignVector<int>(p->perm());

  return option;
}

std::unique_ptr<MeanOptions> Model::MakeMeanOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::MeanOptions*>(
      op->builtin_options());

  std::unique_ptr<MeanOptions> option = std::make_unique<MeanOptions>();

  option->keep_dims = p->keep_dims();
  option->axis = AssignVector<int>(p->axis());

  return option;
}

std::unique_ptr<SqueezeOptions> Model::MakeSqueezeOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SqueezeOptions*>(
      op->builtin_options());

  std::unique_ptr<SqueezeOptions> option = std::make_unique<SqueezeOptions>();

  option->squeeze_dims = AssignVector<int>(p->squeeze_dims());

  return option;
}

std::unique_ptr<BuiltinOptions> Model::HandleBuiltinOptions(
    const tflite::Operator* op) {
  auto op_type = op->builtin_options_type();

  switch (op_type) {
    case tflite::BuiltinOptions::Conv2DOptions:
      return MakeConv2DOptions(op);
      break;

    case tflite::BuiltinOptions::DepthwiseConv2DOptions:
      return MakeDepthwiseConv2DOptions(op);
      break;

    case tflite::BuiltinOptions::ConcatEmbeddingsOptions:
      return MakeConcatEmbeddingsOptions(op);
      break;

    case tflite::BuiltinOptions::LSHProjectionOptions:
      return MakeLSHProjectionOptions(op);
      break;

    case tflite::BuiltinOptions::Pool2DOptions:
      return MakePool2DOptions(op);
      break;

    case tflite::BuiltinOptions::SVDFOptions:
      return MakeSVDFOptions(op);
      break;

    case tflite::BuiltinOptions::RNNOptions:
      return MakeRNNOptions(op);
      break;

    case tflite::BuiltinOptions::FullyConnectedOptions:
      return MakeFullyConnectedOptions(op);
      break;

    case tflite::BuiltinOptions::SoftmaxOptions:
      return MakeSoftmaxOptions(op);
      break;

    case tflite::BuiltinOptions::ConcatenationOptions:
      return MakeConcatenationOptions(op);
      break;

    case tflite::BuiltinOptions::AddOptions:
      return MakeAddOptions(op);
      break;

    case tflite::BuiltinOptions::L2NormOptions:
      return MakeL2NormOptions(op);
      break;

    case tflite::BuiltinOptions::LocalResponseNormalizationOptions:
      return MakeLocalResponseNormalizationOptions(op);
      break;

    case tflite::BuiltinOptions::LSTMOptions:
      return MakeLSTMOptions(op);
      break;

    case tflite::BuiltinOptions::ResizeBilinearOptions:
      return MakeResizeBilinearOptions(op);
      break;

    case tflite::BuiltinOptions::CallOptions:
      return MakeCallOptions(op);
      break;

    case tflite::BuiltinOptions::ReshapeOptions:
      return MakeReshapeOptions(op);
      break;

    case tflite::BuiltinOptions::SkipGramOptions:
      return MakeSkipGramOptions(op);
      break;

    case tflite::BuiltinOptions::SpaceToDepthOptions:
      return MakeSpaceToDepthOptions(op);
      break;

    case tflite::BuiltinOptions::EmbeddingLookupSparseOptions:
      return MakeEmbeddingLookupSparseOptions(op);
      break;

    case tflite::BuiltinOptions::MulOptions:
      return MakeMulOptions(op);
      break;

    case tflite::BuiltinOptions::PadOptions:
      return MakePadOptions(op);
      break;

    case tflite::BuiltinOptions::GatherOptions:
      return MakeGatherOptions(op);
      break;

    case tflite::BuiltinOptions::BatchToSpaceNDOptions:
      return MakeBatchToSpaceNDOptions(op);
      break;

    case tflite::BuiltinOptions::SpaceToBatchNDOptions:
      return MakeSpaceToBatchNDOptions(op);
      break;

    case tflite::BuiltinOptions::TransposeOptions:
      return MakeTransposeOptions(op);
      break;

    case tflite::BuiltinOptions::MeanOptions:
      return MakeMeanOptions(op);
      break;

    case tflite::BuiltinOptions::SubOptions:
      return MakeSubOptions(op);
      break;

    case tflite::BuiltinOptions::DivOptions:
      return MakeDivOptions(op);
      break;

    case tflite::BuiltinOptions::SqueezeOptions:
      return MakeSqueezeOptions(op);
      break;

    case tflite::BuiltinOptions::SequenceRNNOptions:
      return MakeSequenceRNNOptions(op);
      break;

    default:
      return MakeNoneOptions(op);
  }
}

void Model::PopulateGraphOperators(const tflite::SubGraph* graph) {
  auto operators = graph->operators();
  std::vector<Operator> vec_operators;

  // get operators
  for (auto it = operators->begin(); it != operators->end(); ++it) {
    std::vector<int> vec_ins = AssignVector<int>(it->inputs());
    std::vector<int> vec_outs = AssignVector<int>(it->outputs());

    std::string opt_str = tflite::EnumNamesBuiltinOptions()[static_cast<int>(
        it->builtin_options_type())];

    // get builtin options
    std::unique_ptr<BuiltinOptions> builtin_op(HandleBuiltinOptions(*it));

    graph_.AddOperator(Operator(it->opcode_index(), std::move(builtin_op),
        opt_str, std::move(vec_ins), std::move(vec_outs)));
  }
}

void Model::PopulateGraph() {
  auto graph = fb_model_->subgraphs()->Get(0);

  PopulateGraphInputs(graph);
  PopulateGraphOutputs(graph);
  PopulateGraphTensors(graph);
  PopulateGraphOperators(graph);
}

void Model::PopulateBuffers() {
  auto buffer_vec = fb_model_->buffers();

  if (!buffer_vec) {
    return;
  }

  for (auto it = buffer_vec->begin(); it != buffer_vec->end(); ++it) {
    std::vector<u_char> buf = AssignVector<u_char>(it->data());
    buffers_.push_back(std::move(buf));
  }
}

const char* Model::description() {
  return fb_model_->description()->c_str();
}

}
