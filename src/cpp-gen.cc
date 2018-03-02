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

  ss << "dimensions_" << count << "[] = " << dimensions << ";\n";
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
  ss << "  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG,\n"
  ss << "      \"" << boost::str(msg) << "\");";
  ss << "  return false;";
  ss << "}"

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
    ss << CheckStatus("ANeuralNetworksModel_addOperand failed for operand %1%"
        %count);

    // insert operand value
    ss << "status = ANeuralNetworksModel_setOperandValueFromMemory(model, ";
    ss << count << ", memory_model, offset, tensor_size);"
    ss << CheckStatus("ANeuralNetworksModel_setOperandValueFromMemory "
        "failed for operand %1%"%count);
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
