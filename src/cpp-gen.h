#ifndef ANNC_CCP_GEN_H
#define ANNC_CCP_GEN_H

#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "model.h"

namespace annc {

class TensorsHeader {
 public:
  TensorsHeader(Model& model): model_(model) {}

  std::string Assembler();

 private:
  std::string Generate();

  Model& model_;
};

class ModelGen {
 public:
  ModelGen(Model& model): model_(model) {}

 private:
  std::string Generate();
  std::string GenerateTensorType(const Tensor& tensor, int count);
  std::string GenerateTensorsCode();
  std::string TensorTypeStr(TensorType type);
  std::string TensorDim(const std::vector<int>& dim);
  float TensorQuantizationScale(const QuantizationParameters& q);
  int TensorQuantizationZeroPoint(const QuantizationParameters& q);
  std::string CheckStatus(const boost::format& msg);

  Model& model_;
};

class CppGen {
 public:
  CppGen(Model& model): model_(model) {}

  void GenFiles(const std::vector<std::string>& namespace_vec,
      const boost::filesystem::path& path);

 private:
  Model& model_;
};

}

#endif  // ANNC_CCP_GEN_H
