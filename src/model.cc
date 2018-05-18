#include "model.h"

#include <cstdio>
#include <iostream>

#include "exception.h"

namespace nnt {

FlatBufferModel::FlatBufferModel(const std::string& fname) {
  FILE* file = fopen(fname.c_str(), "rb");
  fseek(file, 0L, SEEK_END);
  int length = ftell(file);
  fseek(file, 0L, SEEK_SET);
  data_ = new char[length];
  fread(data_, sizeof(char), length, file);
  fclose(file);
  len_ = length;
}

FlatBufferModel::~FlatBufferModel() {
  delete data_;
}

char* FlatBufferModel::data() {
  return data_;
}

int FlatBufferModel::Length() const {
  return len_;
}

Model::Model(const std::string& fname)
  : flat_buffers_(fname)
  , fb_model_(tflite::GetModel(flat_buffers_.data())) {
  PopulateBuffers();
  PopulateOperatorsCode();
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
    case tflite::TensorType_FLOAT32:
      return TensorType::FLOAT32;
      break;

    case tflite::TensorType_FLOAT16:
      return TensorType::FLOAT16;
      break;

    case tflite::TensorType_INT32:
      return TensorType::INT32;
      break;

    case tflite::TensorType_UINT8:
      return TensorType::UINT8;
      break;

    case tflite::TensorType_INT64:
      return TensorType::INT64;
      break;

    case tflite::TensorType_STRING:
      return TensorType::STRING;
      break;
#ifdef NEWER_TENSORFLOW
    case tflite::TensorType_BOOL:
      return TensorType::BOOL;
      break;
#endif
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
    case tflite::ActivationFunctionType_NONE:
      return ActivationFunctionType::NONE;
      break;

    case tflite::ActivationFunctionType_RELU:
      return ActivationFunctionType::RELU;
      break;

    case tflite::ActivationFunctionType_RELU_N1_TO_1:
      return ActivationFunctionType::NONE;
      break;

    case tflite::ActivationFunctionType_RELU6:
      return ActivationFunctionType::RELU6;
      break;

    case tflite::ActivationFunctionType_TANH:
      return ActivationFunctionType::TANH;
      break;

    case tflite::ActivationFunctionType_SIGN_BIT:
      return ActivationFunctionType::SIGN_BIT;
      break;

    default:
      return ActivationFunctionType::NONE;
  }
}

Padding Model::ConvertPadding(tflite::Padding padding) {
  if (padding == tflite::Padding_SAME) {
    return Padding::SAME;
  } else if (padding == tflite::Padding_VALID) {
    return Padding::VALID;
  } else {
    return Padding::UNKNOWN;
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
#ifdef NEWER_TENSORFLOW
  option->dilation_w_factor = p->dilation_w_factor();
  option->dilation_h_factor = p->dilation_h_factor();
#endif
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
    case tflite::LSHProjectionType_UNKNOWN:
      option->type = LSHProjectionType::UNKNOWN;
      break;

    case tflite::LSHProjectionType_SPARSE:
      option->type = LSHProjectionType::SPARSE;
      break;

    case tflite::LSHProjectionType_DENSE:
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

  option->align_corners = p->align_corners();

  return option;
}

std::unique_ptr<CallOptions> Model::MakeCallOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::CallOptions*>(op->builtin_options());

  std::unique_ptr<CallOptions> option = std::make_unique<CallOptions>();

  option->subgraph = p->subgraph();

  return option;
}

std::unique_ptr<PadOptions> Model::MakePadOptions(const tflite::Operator*) {
  std::unique_ptr<PadOptions> option = std::make_unique<PadOptions>();
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
    const tflite::Operator*) {
  std::unique_ptr<SpaceToBatchNDOptions> option =
      std::make_unique<SpaceToBatchNDOptions>();

  return option;
}

std::unique_ptr<BatchToSpaceNDOptions> Model::MakeBatchToSpaceNDOptions(
    const tflite::Operator*) {
  std::unique_ptr<BatchToSpaceNDOptions> option =
      std::make_unique<BatchToSpaceNDOptions>();

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
    case tflite::CombinerType_SUM:
      option->combiner = CombinerType::SUM;
      break;

    case tflite::CombinerType_MEAN:
      option->combiner = CombinerType::MEAN;
      break;

    case tflite::CombinerType_SQRTN:
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
    const tflite::Operator*) {
  std::unique_ptr<TransposeOptions> option =
      std::make_unique<TransposeOptions>();

  return option;
}

std::unique_ptr<MeanOptions> Model::MakeMeanOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::MeanOptions*>(
      op->builtin_options());

  std::unique_ptr<MeanOptions> option = std::make_unique<MeanOptions>();

  option->keep_dims = p->keep_dims();

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

std::unique_ptr<ExpOptions> Model::MakeExpOptions(
    const tflite::Operator*) {
  std::unique_ptr<ExpOptions> option = std::make_unique<ExpOptions>();

  return option;
}

std::unique_ptr<TopKV2Options> Model::MakeTopKV2Options(
    const tflite::Operator*) {
  std::unique_ptr<TopKV2Options> option = std::make_unique<TopKV2Options>();
  return option;
}

std::unique_ptr<SplitOptions> Model::MakeSplitOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::SplitOptions*>(
      op->builtin_options());

  std::unique_ptr<SplitOptions> option = std::make_unique<SplitOptions>();

  option->num_splits = p->num_splits();

  return option;
}

std::unique_ptr<LogSoftmaxOptions> Model::MakeLogSoftmaxOptions(
    const tflite::Operator*) {
  std::unique_ptr<LogSoftmaxOptions> option =
      std::make_unique<LogSoftmaxOptions>();

  return option;
}

std::unique_ptr<CastOptions> Model::MakeCastOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::CastOptions*>(
      op->builtin_options());

  std::unique_ptr<CastOptions> option = std::make_unique<CastOptions>();

  option->in_data_type = ConvertTensorType(p->in_data_type());
  option->out_data_type = ConvertTensorType(p->out_data_type());

  return option;
}

std::unique_ptr<DequantizeOptions> Model::MakeDequantizeOptions(
    const tflite::Operator*) {
  std::unique_ptr<DequantizeOptions> option =
      std::make_unique<DequantizeOptions>();

  return option;
}

#ifdef NEWER_TENSORFLOW
std::unique_ptr<MaximumMinimumOptions> Model::MakeMaximumMinimumOptions(
    const tflite::Operator*) {
  std::unique_ptr<MaximumMinimumOptions> option =
      std::make_unique<MaximumMinimumOptions>();

  return option;
}

std::unique_ptr<ArgMaxOptions> Model::MakeArgMaxOptions(
    const tflite::Operator* op) {
  auto p = reinterpret_cast<const tflite::ArgMaxOptions*>(
      op->builtin_options());

  std::unique_ptr<ArgMaxOptions> option = std::make_unique<ArgMaxOptions>();

  option->output_type = ConvertTensorType(p->output_type());

  return option;
}

std::unique_ptr<LessOptions> Model::MakeLessOptions(
    const tflite::Operator*) {
  std::unique_ptr<LessOptions> option = std::make_unique<LessOptions>();
  return option;
}

std::unique_ptr<NegOptions> Model::MakeNegOptions(
    const tflite::Operator*) {
  std::unique_ptr<NegOptions> option = std::make_unique<NegOptions>();
  return option;
}
#else
std::unique_ptr<MaximumOptions> Model::MakeMaximumOptions(
    const tflite::Operator*) {
  std::unique_ptr<MaximumOptions> option = std::make_unique<MaximumOptions>();

  return option;
}
#endif

std::unique_ptr<BuiltinOptions> Model::HandleBuiltinOptions(
    const tflite::Operator* op) {
  auto op_type = op->builtin_options_type();

  switch (op_type) {
    case tflite::BuiltinOptions_Conv2DOptions:
      return MakeConv2DOptions(op);
      break;

    case tflite::BuiltinOptions_DepthwiseConv2DOptions:
      return MakeDepthwiseConv2DOptions(op);
      break;

    case tflite::BuiltinOptions_ConcatEmbeddingsOptions:
      return MakeConcatEmbeddingsOptions(op);
      break;

    case tflite::BuiltinOptions_LSHProjectionOptions:
      return MakeLSHProjectionOptions(op);
      break;

    case tflite::BuiltinOptions_Pool2DOptions:
      return MakePool2DOptions(op);
      break;

    case tflite::BuiltinOptions_SVDFOptions:
      return MakeSVDFOptions(op);
      break;

    case tflite::BuiltinOptions_RNNOptions:
      return MakeRNNOptions(op);
      break;

    case tflite::BuiltinOptions_FullyConnectedOptions:
      return MakeFullyConnectedOptions(op);
      break;

    case tflite::BuiltinOptions_SoftmaxOptions:
      return MakeSoftmaxOptions(op);
      break;

    case tflite::BuiltinOptions_ConcatenationOptions:
      return MakeConcatenationOptions(op);
      break;

    case tflite::BuiltinOptions_AddOptions:
      return MakeAddOptions(op);
      break;

    case tflite::BuiltinOptions_L2NormOptions:
      return MakeL2NormOptions(op);
      break;

    case tflite::BuiltinOptions_LocalResponseNormalizationOptions:
      return MakeLocalResponseNormalizationOptions(op);
      break;

    case tflite::BuiltinOptions_LSTMOptions:
      return MakeLSTMOptions(op);
      break;

    case tflite::BuiltinOptions_ResizeBilinearOptions:
      return MakeResizeBilinearOptions(op);
      break;

    case tflite::BuiltinOptions_CallOptions:
      return MakeCallOptions(op);
      break;

    case tflite::BuiltinOptions_ReshapeOptions:
      return MakeReshapeOptions(op);
      break;

    case tflite::BuiltinOptions_SkipGramOptions:
      return MakeSkipGramOptions(op);
      break;

    case tflite::BuiltinOptions_SpaceToDepthOptions:
      return MakeSpaceToDepthOptions(op);
      break;

    case tflite::BuiltinOptions_EmbeddingLookupSparseOptions:
      return MakeEmbeddingLookupSparseOptions(op);
      break;

    case tflite::BuiltinOptions_MulOptions:
      return MakeMulOptions(op);
      break;

    case tflite::BuiltinOptions_PadOptions:
      return MakePadOptions(op);
      break;

    case tflite::BuiltinOptions_GatherOptions:
      return MakeGatherOptions(op);
      break;

    case tflite::BuiltinOptions_BatchToSpaceNDOptions:
      return MakeBatchToSpaceNDOptions(op);
      break;

    case tflite::BuiltinOptions_SpaceToBatchNDOptions:
      return MakeSpaceToBatchNDOptions(op);
      break;

    case tflite::BuiltinOptions_TransposeOptions:
      return MakeTransposeOptions(op);
      break;

    case tflite::BuiltinOptions_MeanOptions:
      return MakeMeanOptions(op);
      break;

    case tflite::BuiltinOptions_SubOptions:
      return MakeSubOptions(op);
      break;

    case tflite::BuiltinOptions_DivOptions:
      return MakeDivOptions(op);
      break;

    case tflite::BuiltinOptions_SqueezeOptions:
      return MakeSqueezeOptions(op);
      break;

    case tflite::BuiltinOptions_SequenceRNNOptions:
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

    // get the operator code reference given the index o operator table
    size_t opcode_index = static_cast<size_t>(it->opcode_index());
    const OperatorCode& op_code = operators_code_[opcode_index];

    graph_.AddOperator(Operator(opcode_index, op_code, std::move(builtin_op),
        opt_str, std::move(vec_ins), std::move(vec_outs)));
  }
}

void Model::PopulateGraph() {
  if (flat_buffers_.Length() == 0) {
    FATAL("Model file is empty")
    return;
  }

  auto subgraphs = fb_model_->subgraphs();
  if (!subgraphs) {
    FATAL("No subgraph found")
    return;
  }

  auto graph = subgraphs->Get(0);

  PopulateGraphInputs(graph);
  PopulateGraphOutputs(graph);
  PopulateGraphTensors(graph);
  PopulateGraphOperators(graph);
}

void Model::PopulateBuffers() {
  auto buffer_vec = fb_model_->buffers();

  // test if buffer_vec is null to avoid crash on flatbuffers
  if (!buffer_vec) {
    return;
  }

  for (auto it = buffer_vec->begin(); it != buffer_vec->end(); ++it) {
    std::vector<u_char> buf = AssignVector<u_char>(it->data());
    buffers_.push_back(std::move(buf));
  }
}

BuiltinOperator Model::ConvertOperatorCode(tflite::BuiltinOperator type) {
  switch (type) {
    case tflite::BuiltinOperator_ADD:
      return BuiltinOperator::ADD;
      break;

    case tflite::BuiltinOperator_AVERAGE_POOL_2D:
      return BuiltinOperator::AVERAGE_POOL_2D;
      break;

    case tflite::BuiltinOperator_CONCATENATION:
      return BuiltinOperator::CONCATENATION;
      break;

    case tflite::BuiltinOperator_CONV_2D:
      return BuiltinOperator::CONV_2D;
      break;

    case tflite::BuiltinOperator_DEPTHWISE_CONV_2D:
      return BuiltinOperator::DEPTHWISE_CONV_2D;
      break;

    case tflite::BuiltinOperator_DEQUANTIZE:
      return BuiltinOperator::DEQUANTIZE;
      break;

    case tflite::BuiltinOperator_EMBEDDING_LOOKUP:
      return BuiltinOperator::EMBEDDING_LOOKUP;
      break;
#ifdef NEWER_TENSORFLOW
    case tflite::BuiltinOperator_FLOOR:
      return BuiltinOperator::FLOOR;
      break;
#endif
    case tflite::BuiltinOperator_FULLY_CONNECTED:
      return BuiltinOperator::FULLY_CONNECTED;
      break;

    case tflite::BuiltinOperator_HASHTABLE_LOOKUP:
      return BuiltinOperator::HASHTABLE_LOOKUP;
      break;

    case tflite::BuiltinOperator_L2_NORMALIZATION:
      return BuiltinOperator::L2_NORMALIZATION;
      break;

    case tflite::BuiltinOperator_L2_POOL_2D:
      return BuiltinOperator::L2_POOL_2D;
      break;

    case tflite::BuiltinOperator_LOCAL_RESPONSE_NORMALIZATION:
      return BuiltinOperator::LOCAL_RESPONSE_NORMALIZATION;
      break;

    case tflite::BuiltinOperator_LOGISTIC:
      return BuiltinOperator::LOGISTIC;
      break;

    case tflite::BuiltinOperator_LSH_PROJECTION:
      return BuiltinOperator::LSH_PROJECTION;
      break;

    case tflite::BuiltinOperator_LSTM:
      return BuiltinOperator::LSTM;
      break;

    case tflite::BuiltinOperator_MAX_POOL_2D:
      return BuiltinOperator::MAX_POOL_2D;
      break;

    case tflite::BuiltinOperator_MUL:
      return BuiltinOperator::MUL;
      break;

    case tflite::BuiltinOperator_RELU:
      return BuiltinOperator::RELU;
      break;

    case tflite::BuiltinOperator_RELU_N1_TO_1:
      return BuiltinOperator::RELU1;
      break;

    case tflite::BuiltinOperator_RELU6:
      return BuiltinOperator::RELU6;
      break;

    case tflite::BuiltinOperator_RESHAPE:
      return BuiltinOperator::RESHAPE;
      break;

    case tflite::BuiltinOperator_RNN:
      return BuiltinOperator::RNN;
      break;

    case tflite::BuiltinOperator_SOFTMAX:
      return BuiltinOperator::SOFTMAX;
      break;

    case tflite::BuiltinOperator_SPACE_TO_DEPTH:
      return BuiltinOperator::SPACE_TO_DEPTH;
      break;

    case tflite::BuiltinOperator_SVDF:
      return BuiltinOperator::SVDF;
      break;

    case tflite::BuiltinOperator_TANH:
      return BuiltinOperator::TANH;
      break;

    case tflite::BuiltinOperator_CONCAT_EMBEDDINGS:
      return BuiltinOperator::CONCAT_EMBEDDINGS;
      break;

    case tflite::BuiltinOperator_SKIP_GRAM:
      return BuiltinOperator::SKIP_GRAM;
      break;

    case tflite::BuiltinOperator_CALL:
      return BuiltinOperator::CALL;
      break;

    case tflite::BuiltinOperator_CUSTOM:
      return BuiltinOperator::CUSTOM;
      break;

    case tflite::BuiltinOperator_EMBEDDING_LOOKUP_SPARSE:
      return BuiltinOperator::EMBEDDING_LOOKUP_SPARSE;
      break;

    case tflite::BuiltinOperator_PAD:
      return BuiltinOperator::PAD;
      break;

    case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_RNN:
      return BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_RNN;
      break;

    case tflite::BuiltinOperator_GATHER:
      return BuiltinOperator::GATHER;
      break;

    case tflite::BuiltinOperator_BATCH_TO_SPACE_ND:
      return BuiltinOperator::BATCH_TO_SPACE_ND;
      break;

    case tflite::BuiltinOperator_SPACE_TO_BATCH_ND:
      return BuiltinOperator::SPACE_TO_BATCH_ND;
      break;

    case tflite::BuiltinOperator_TRANSPOSE:
      return BuiltinOperator::TRANSPOSE;
      break;

    case tflite::BuiltinOperator_MEAN:
      return BuiltinOperator::MEAN;
      break;

    case tflite::BuiltinOperator_SUB:
      return BuiltinOperator::SUB;
      break;

    case tflite::BuiltinOperator_DIV:
      return BuiltinOperator::DIV;
      break;

    case tflite::BuiltinOperator_SQUEEZE:
      return BuiltinOperator::SQUEEZE;
      break;

    case tflite::BuiltinOperator_UNIDIRECTIONAL_SEQUENCE_LSTM:
      return BuiltinOperator::UNIDIRECTIONAL_SEQUENCE_LSTM;
      break;

    case tflite::BuiltinOperator_STRIDED_SLICE:
      return BuiltinOperator::STRIDED_SLICE;
      break;

    case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_RNN:
      return BuiltinOperator::BIDIRECTIONAL_SEQUENCE_RNN;
      break;

    case tflite::BuiltinOperator_EXP:
      return BuiltinOperator::EXP;
      break;

    case tflite::BuiltinOperator_TOPK_V2:
      return BuiltinOperator::TOPK_V2;
      break;

    case tflite::BuiltinOperator_SPLIT:
      return BuiltinOperator::SPLIT;
      break;

    case tflite::BuiltinOperator_LOG_SOFTMAX:
      return BuiltinOperator::LOG_SOFTMAX;
      break;

    case tflite::BuiltinOperator_DELEGATE:
      return BuiltinOperator::DELEGATE;
      break;

    case tflite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM:
      return BuiltinOperator::BIDIRECTIONAL_SEQUENCE_LSTM;
      break;

    case tflite::BuiltinOperator_CAST:
      return BuiltinOperator::CAST;
      break;

    case tflite::BuiltinOperator_PRELU:
      return BuiltinOperator::PRELU;
      break;

    case tflite::BuiltinOperator_MAXIMUM:
      return BuiltinOperator::MAXIMUM;
      break;
#ifdef NEWER_TENSORFLOW
    case tflite::BuiltinOperator_ARG_MAX:
      return BuiltinOperator::ARG_MAX;
      break;

    case tflite::BuiltinOperator_MINIMUM:
      return BuiltinOperator::MINIMUM;
      break;

    case tflite::BuiltinOperator_LESS:
      return BuiltinOperator::LESS;
      break;

    case tflite::BuiltinOperator_NEG:
      return BuiltinOperator::NEG;
      break;
#endif
    default:
      return BuiltinOperator::NONE;
  }
}

void Model::PopulateOperatorsCode() {
  auto op_codes_vec = fb_model_->operator_codes();

  if (!op_codes_vec) {
    return;
  }

  for (auto it = op_codes_vec->begin(); it != op_codes_vec->end(); ++it) {
    auto custom_code = it->custom_code();

    OperatorCode op_code {
      .builtin_code = ConvertOperatorCode(it->builtin_code()),
      .custom_code = custom_code? "\"" + custom_code->str() +"\"" : "\"\""
    };

    operators_code_.push_back(std::move(op_code));
  }
}

const char* Model::description() {
  return fb_model_->description()->c_str();
}

}
