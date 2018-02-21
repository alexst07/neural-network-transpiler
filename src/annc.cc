#include <iostream>

#include "model.h"
#include "dump.h"

int main(int argc, char **argv) {
  std::string fname = argv[1];

  annc::Model model(fname);

  annc::DumpGraph dump(model);
  dump.Print();
}
