#include <iostream>
#include <boost/program_options.hpp>

#include "model.h"
#include "cpp-gen.h"
#include "dump.h"
#include "exception.h"

void GenerateJniFiles(const std::string& str_model, const std::string& str_path,
    const std::string& java_package) {
  annc::Model model(str_model);
  annc::CppGen cpp(model);
  boost::filesystem::path path(str_path);
  cpp.GenFiles(path, java_package);
  std::cout << "Finish!\n";
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  std::string str_path;
  std::string java_package;
  std::string str_model;

  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("model,m", po::value<std::string>(), "flatbuffer neural network model")
      ("path,p", po::value<std::string>(), "store generated files on this path")
      ("javapackage,j", po::value<std::string>(), "java package for JNI");

    po::variables_map vm;
    po::store(parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << '\n';
      return 0;
    }

    if (!vm.count("model")) {
      std::cerr << "--model must not be empty" << '\n';
      std::cerr << desc << '\n';
      return 0;
    }

    if (vm.count("path")) {
      str_path = vm["path"].as<std::string>();
    } else {
      str_path = "./";
    }

    if (!vm.count("javapackage")) {
      std::cerr << "--javapackage must not be empty" << '\n';
      std::cerr << desc << '\n';
      return 0;
    }

    str_model = vm["model"].as<std::string>();
    java_package = vm["javapackage"].as<std::string>();

    GenerateJniFiles(str_model, str_path, java_package);
  } catch (const boost::program_options::error &e) {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (const annc::Exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}
