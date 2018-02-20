#include <iostream>
#include <sstream>

#include "dump.h"

namespace annc {

void DumpGraph::Print() {
  Graph& graph = model_.graph();

  std::cout << "Inputs:";
  for (const auto& i : graph.Inputs()) {
    std::cout << " " << i;
  }
  std::cout << "\n";

  std::cout << "Outputs:";
  for (const auto& i : graph.Outputs()) {
    std::cout << " " << i;
  }
  std::cout << "\n";

  std::cout << "\nTensors:\n";
  int count = 0;
  for (const auto& tensor: graph.Tensors()) {
    std::cout << "[" << count++ << "] ";
    std::cout << "name: " << tensor.name() << " [ ";
    for (const auto&i : tensor.shape()) {
      std::cout << i << " ";
    }
    std::cout << "]\n";
  }

  std::cout << "\nOperators:\n";
  for (const auto& op: graph.Operators()) {
    std::cout << "index: " << op.index() << ", ";
    std::cout << "builtin_op: " << op.builtin_op_str() << ", ";

    std::cout << "inputs:";
    for (const auto& i : op.inputs()) {
      std::cout << " " << i;
    }
    std::cout << ", ";

    std::cout << "outputs:";
    for (const auto& i : op.outputs()) {
      std::cout << " " << i;
    }
    std::cout << "\n";
  }
}

std::string DumpGraph::FormatTensorName(const std::string& name) {
  size_t pos = name.find_last_of('/');

  if (pos != std::string::npos) {
    return name.substr(pos);
  }

  return name;
}

std::string DumpGraph::Dot() {
  std::stringstream ss;

  Graph& graph = model_.graph();

  ss << "digraph {\n";

  int count = 0;
  for (const auto& tensor: graph.Tensors()) {
    ss << "  T" << count++ << " [";
    ss << "shape=box label=\"" << "..." << FormatTensorName(tensor.name())
       << "\"]\n";
  }

  count = 0;
  for (const auto& op: graph.Operators()) {
    ss << "  O" << count++ << " [";
    ss << "label=\"" << op.builtin_op_str() << "\"]\n";
  }

  count = 0;
  for (const auto& op: graph.Operators()) {
    for (const auto& i : op.inputs()) {
      ss << "  T" << i << " -> " << "O" << count << "\n";
    }

    for (const auto& i : op.outputs()) {
      ss << "  O" << count << " -> " << "T" << i << "\n";
    }

    ++count;
  }

  ss << "}\n";

  return ss.str();
}

}
