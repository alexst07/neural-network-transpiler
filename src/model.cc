#include <cstdio>
#include <iostream>

#include "model.h"

namespace annc {

FlatBufferModel::FlatBufferModel(const std::string& fname) {
  FILE* file = fopen(fname.c_str(), "rb");
  fseek(file, 0L, SEEK_END);
  int length = ftell(file);
  fseek(file, 0L, SEEK_SET);
  data_ = new char[length];
  fread(data_, sizeof(char), length, file);
  fclose(file);
}

FlatBufferModel::~FlatBufferModel() {
  delete data_;
}

char* FlatBufferModel::data() {
  return data_;
}

Model::Model(const std::string& fname)
  : flat_buffers_(fname)
  , fb_model_(tflite::GetModel(flat_buffers_.data())) {
  PopulateBuffers();
  PopulateGraph();
}

void Model::PopulateGraphInputs(const tflite::SubGraph* graph) {
  auto graph_inputs = graph->inputs();

  // get inputs
  std::vector<int> inputs;
  for (auto it = graph_inputs->begin(); it != graph_inputs->end(); ++it) {
    inputs.push_back(*it);
  }

  graph_.SetInputs(std::move(inputs));
}

void Model::PopulateGraphOutputs(const tflite::SubGraph* graph) {
  auto graph_outputs = graph->outputs();

  // get outputs
  std::vector<int> outputs;
  for (auto it = graph_outputs->begin(); it != graph_outputs->end(); ++it) {
    outputs.push_back(*it);
  }

  graph_.SetOutputs(std::move(outputs));
}

void Model::PopulateGraphTensors(const tflite::SubGraph* graph) {
  auto tensors = graph->tensors();

  // get tensors
  for (auto it = tensors->begin(); it != tensors->end(); ++it) {
    auto shape = it->shape();
    std::vector<int> vec_shape;

    for (auto it_shape = shape->begin(); it_shape != shape->end(); ++it_shape) {
      vec_shape.push_back(*it_shape);
    }

    std::string name = it->name()->c_str();
    uint buf_index = it->buffer();
    const Buffer& buffer = buffers_[buf_index];
    graph_.AddTensor(std::move(Tensor(std::move(vec_shape), name, buffer)));
  }
}

void Model::PopulateGraphOperators(const tflite::SubGraph* graph) {
  auto operators = graph->operators();
  std::vector<Operator> vec_operators;

  // get operators
  for (auto it = operators->begin(); it != operators->end(); ++it) {
    auto ins = it->inputs();
    std::vector<int> vec_ins;
    for (auto it_ins = ins->begin(); it_ins != ins->end(); ++it_ins) {
      vec_ins.push_back(*it_ins);
    }

    auto outs = it->outputs();
    std::vector<int> vec_outs;
    for (auto it_outs = outs->begin(); it_outs != outs->end(); ++it_outs) {
      vec_outs.push_back(*it_outs);
    }

    std::string opt_str = tflite::EnumNamesBuiltinOptions()[static_cast<int>(
        it->builtin_options_type())];

    graph_.AddOperator(Operator(it->opcode_index(), opt_str,
        std::move(vec_ins), std::move(vec_outs)));
  }
}

void Model::PopulateGraph() {
  auto graph = fb_model_->subgraphs()->Get(0);

  PopulateGraphInputs(graph);
  PopulateGraphOutputs(graph);
  PopulateGraphTensors(graph);
  PopulateGraphOperators(graph);
}

void Model::PopulateBuffers() {
  auto buffer_vec = fb_model_->buffers();

  for (auto it = buffer_vec->begin(); it != buffer_vec->end(); ++it) {
    auto buffer = it->data();

    // if buffer is empty, push back an empty vector to buffers_
    // TODO: Verify if empyt vector must be added
    if (buffer == nullptr) {
      std::vector<u_char> buf;
      buffers_.push_back(std::move(buf));
      continue;
    }

    std::vector<u_char> buf;

    for (auto it_data = buffer->begin(); it_data != buffer->end(); ++it_data) {
      buf.push_back(*it_data);
    }

    buffers_.push_back(std::move(buf));
  }
}

const char* Model::description() {
  return fb_model_->description()->c_str();
}

}
