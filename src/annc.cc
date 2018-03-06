#include <iostream>

#include "model.h"
#include "cpp-gen.h"
#include "dump.h"

int main(int argc, char **argv) {
  std::string fname = argv[1];

  annc::Model model(fname);

  // annc::CppGen cpp(model);
  // std::vector<std::string> namespace_vec = {"test"};
  // boost::filesystem::path path(".");
  // cpp.GenFiles(namespace_vec, path);

  annc::DumpGraph dump(model);
  std::cout << dump.Dot();
}
