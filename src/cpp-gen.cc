#include "cpp-gen.h"

#include <iostream>
#include <sstream>
#include <fstream>

#include "exception.h"

namespace annc {

std::string TensorsHeader::Generate() {
  const std::vector<Buffer>& buffers = model_.Buffers();
  std::string str_buf;

  for (const auto& buf : buffers) {
    if (buf.Data().size() > 0) {
      std::string str_vec = "";
      for (const auto& c : buf.Data()) {
        str_buf += c;
      }
    }
  }

  return str_buf;
}

std::string TensorsHeader::Assembler() {
  return Generate();
}

std::string ModelGen::Generate() {
  std::string str_init = "Init";
}

std::string ModelGen::TensorTypeStr(TensorType type) {
  switch (type) {
    case TensorType::FLOAT32:
      return "ANEURALNETWORKS_TENSOR_FLOAT32";
      break;

    case TensorType::INT32:
      return "ANEURALNETWORKS_TENSOR_INT32";
      break;

    case TensorType::UINT8:
      return "ANEURALNETWORKS_TENSOR_QUANT8_ASYMM";
      break;

    default:
      FATAL("Tensor type not valid for Android NNAPI")
  }
}

std::string ModelGen::TensorDim(const std::vector<int>& dim) {
  std::string str_out = "{";

  for (const auto& e : dim) {
    str_out += std::to_string(e) + ",";
  }

  str_out = str_out.substr(0, str_out.length() - 2);
  str_out += "}";

  return str_out;
}

float ModelGen::TensorQuantizationScale(const QuantizationParameters& q) {
  if (q.scale.size() > 0) {
    return q.scale[0];
  } else {
    return 0.0f;
  }
}

int ModelGen::TensorQuantizationZeroPoint(
    const QuantizationParameters& q) {
  if (q.zero_point.size() > 0) {
    return q.zero_point[0];
  } else {
    return 0;
  }
}

std::string ModelGen::GenerateTensorType(const Tensor& tensor, int count) {
  std::stringstream ss;
  std::string dimensions = TensorDim(tensor.shape());

  ss << "uint32_t dimensions_" << count << "[] = " << dimensions << ";\n";
  ss << "ANeuralNetworksOperandType operand_type_ " << count << "{\n";

  std::string str_tensor_type = TensorTypeStr(tensor.tensor_type());
  int dimension_count = tensor.shape().size();

  float scale;
  int zero_point;

  if (tensor.HasQuantization()) {
    scale = TensorQuantizationScale(tensor.quantization());
    zero_point = TensorQuantizationZeroPoint(tensor.quantization());
  } else {
    scale = 0.0f;
    zero_point = 0;
  }

  ss << ".type = " << str_tensor_type << ",\n";
  ss << ".dimensionCount = " << dimension_count << ",\n";
  ss << ".dimensions = " << "dimensions_" << count << ",\n";
  ss << ".scale = " << scale << "f,\n";
  ss << ".zeroPoint = " << zero_point << "\n";
  ss << "}\n";

  return ss.str();
}

std::string ModelGen::CheckStatus(const boost::format& msg) {
  std::stringstream ss;

  // nnapi always check the result of operation
  ss << "if (status != ANEURALNETWORKS_NO_ERROR) {\n";
  ss << "  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n";
  ss << "      \"" << boost::str(msg) << "\");";
  ss << "  return false;";
  ss << "}";

  return ss.str();
}

std::string ModelGen::GenerateTensorsCode() {
  Graph& graph = model_.graph();
  std::stringstream ss;

  int count = 0;
  for (const auto& tensor: graph.Tensors()) {
    // insert operand type
    ss << GenerateTensorType(tensor, count);

    // insert nnapi operand
    ss << "status = ANeuralNetworksModel_addOperand(model,";
    ss << "operand_type_" << count << ");\n";
    ss << CheckStatus(boost::format("ANeuralNetworksModel_addOperand failed"
        "for operand %1%")%count);

    size_t buf_size = tensor.buffer().Data().size();

    if (buf_size > 0) {
      // get tensor size
      ss << "tensor_size = " << buf_size << ";";

      // insert operand value
      ss << "status = ANeuralNetworksModel_setOperandValueFromMemory(model, ";
      ss << count << ", memory_model, offset, tensor_size);";
      ss << CheckStatus(boost::format(
          "ANeuralNetworksModel_setOperandValueFromMemory "
          "failed for operand %1%")%count);

      // calculates the offset
      ss << "offset += tensor_size";
    }
  }

  return ss.str();
}

std::string ModelGen::GenerateOpInputs(const std::vector<int>& inputs) {
  // inputs loop
  std::string str_in = "";

  for (const auto& in_value : inputs) {
    str_in += " " + std::to_string(in_value) + ",";
  }

  str_in = str_in.substr(0. str_in.length() - 1);
  return str_in;
}

std::string ModelGen::GenerateOpOutputs(const std::vector<int>& outputs) {
  // outputs loop
  std::string str_out = "";

  for (const auto& out_value : outputs) {
    str_out += " " + std::to_string(out_value) + ",";
  }

  str_out = str_out.substr(0. str_out.length() - 1);
  return str_out;
}

std::string ModelGen::OpTypeStr(BuiltinOptionsType type) {
  switch (builtin) {
    case BuiltinOptionsType::BuiltinOperator_ADD:
      return "ANEURALNETWORKS_ADD";
      break;

    case BuiltinOptionsType::AVERAGE_POOL_2D:
      return "ANEURALNETWORKS_AVERAGE_POOL_2D";
      break;

    case BuiltinOptionsType::Pool2DOptions:
      return "ANEURALNETWORKS_MAX_POOL_2D";
      break;

    case BuiltinOptionsType::BuiltinOperator_L2_POOL_2D:
      return "ANEURALNETWORKS_L2_POOL_2D";
      break;

    case BuiltinOptionsType::Conv2DOptions:
      return "ANEURALNETWORKS_CONV_2D";
      break;

    case BuiltinOptionsType::BuiltinOperator_RELU:
      return "ANEURALNETWORKS_RELU";
      break;

    case BuiltinOptionsType::BuiltinOperator_RELU6:
      return "ANEURALNETWORKS_RELU6";
      break;

    case BuiltinOptionsType::BuiltinOperator_TANH:
      return "ANEURALNETWORKS_TANH";
      break;

    case BuiltinOptionsType::BuiltinOperator_LOGISTIC:
      return "ANEURALNETWORKS_LOGISTIC";
      break;

    case BuiltinOptionsType::DepthwiseConv2DOptions:
      return "ANEURALNETWORKS_DEPTHWISE_CONV_2D";
      break;

    case BuiltinOptionsType::ConcatEmbeddingsOptions:
      return "ANEURALNETWORKS_CONCATENATION";
      break;

    case BuiltinOptionsType::BuiltinOperator_SOFTMAX:
      return "ANEURALNETWORKS_SOFTMAX";
      break;

    case BuiltinOptionsType::BuiltinOperator_FULLY_CONNECTED:
      return "ANEURALNETWORKS_FULLY_CONNECTED";
      break;

    case BuiltinOptionsType::BuiltinOperator_RESHAPE:
      return "ANEURALNETWORKS_RESHAPE";
      break;

    case BuiltinOptionsType::BuiltinOperator_SPACE_TO_DEPTH:
      return "ANEURALNETWORKS_SPACE_TO_DEPTH";
      break;

    case BuiltinOptionsType::BuiltinOperator_LSTM:
      return "ANEURALNETWORKS_LSTM";
      break;

    default:
      FATAL(boost::format("Not supported type on NNAPI"))
  }
}

std::string ModelGen::GenerateOpCode() {
  Graph& graph = model_.graph();
  std::stringstream ss;

  int count = 0;
  for (const auto& op: graph.Operators()) {
    ss << "uint32_t input_operands_" << count << "[";
    ss << op.inputs().size() <<"] = {";
    ss << GenerateOpInputs(op.inputs()) << "};\n";

    ss << "uint32_t output_operands_" << count << "[";
    ss << op.outputs().size() <<"] = {";
    ss << GenerateOpOutputs(op.outputs()) << "};\n";

    ss << "status = ANeuralNetworksModel_addOperation(model, ";
    ss << OpTypeStr(op.builtin_op().type) << ", sizeof(input_operands_" ;
    ss << count <<"), input_operands_" << count << ", ";
    ss << ", sizeof(output_operands_" << count << "), ";
    ss << "input_operands_" << count << ");\n";

    ss << CheckStatus(boost::format(
        "ANeuralNetworksModel_addOperation failed for operation %1%")%count);
  }
}

void CppGen::GenFiles(const std::vector<std::string>& namespace_vec,
    const boost::filesystem::path& path) {
  std::ofstream tensors_file(path.string() + "/weights_biases.bin",
      std::ofstream::out | std::ofstream::binary);

  if (!tensors_file.is_open()) {
    FATAL("Fail on create tensors_params.h file")
  }

  TensorsHeader tensor_header(model_);
  std::string buf = tensor_header.Assembler();
  tensors_file.write(buf.c_str(), buf.length());
  tensors_file.close();
}

}
