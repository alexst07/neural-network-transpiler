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

void GenerateDotFile(const std::string& filename,
    const std::string& str_model) {
  annc::Model model(str_model);
  annc::DumpGraph dump(model);

  std::ofstream dot_file(filename, std::ofstream::out);

  if (!dot_file.is_open()) {
    std::cerr << "Fail on create dot file: '" << filename << "'\n";
    return;
  }

  std::string dot_src = dump.Dot();
  dot_file.write(dot_src.c_str(), dot_src.length());
  dot_file.close();

  std::cout << "Dot file: '" << filename << "' generated.\n";
}

void Info(const std::string& str_model) {
  annc::Model model(str_model);
  annc::DumpGraph dump(model);
  std::cout << dump.Info();
}

int main(int argc, char **argv) {
  namespace po = boost::program_options;
  std::string str_path;
  std::string java_package;
  std::string str_model;
  std::string str_dot;
  bool flag_info;

  try {
    po::options_description desc{"Options"};
    desc.add_options()
      ("help,h", "Help screen")
      ("info,i", po::bool_switch(&flag_info), "Info about model")
      ("dot,d", po::value<std::string>(), "Generate dot file")
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

    str_model = vm["model"].as<std::string>();

    if (flag_info) {
      Info(str_model);
      return 0;
    }

    if (vm.count("dot")) {
      str_dot = vm["dot"].as<std::string>();
      GenerateDotFile(str_dot, str_model);
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

    java_package = vm["javapackage"].as<std::string>();

    GenerateJniFiles(str_model, str_path, java_package);
  } catch (const boost::program_options::error &e) {
    std::cerr << "Error: " << e.what() << '\n';
  } catch (const annc::Exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
  }
}
