#ifndef NNT_DUMP_H
#define NNT_DUMP_H

#include <string>
#include <vector>

#include "model.h"

namespace nnt {

class DumpGraph {
 public:
  DumpGraph(Model& model): model_(model) {}

  void Print();

  std::string TensorShape(const Tensor& tensor);

  std::string Dot();

  std::string Info();

  std::string TensorType(const Tensor& tensor);

 private:
  std::string FormatTensorName(const std::string& name);

  Model& model_;
};

}

#endif
