#pragma once
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "graph/ComputationGraph.h"

namespace nnet {
namespace engine {
using graph::Tensor;

inline Eigen::Map<const Eigen::ArrayXi> castToEigenIArray1D(const Tensor& tensor) {
  return Eigen::Map<const Eigen::ArrayXi>((int*)tensor.buffer_->get(), tensor.attr_->dims_[0]);
}

class Engine {
 public:
  using NameMappingFN = std::function<std::unique_ptr<Tensor>(const std::string& name)>;
  Engine(const graph::Graph& graph) : graph_{graph} {}
  virtual ~Engine() {}

  virtual void randomize(NameMappingFN fn = nullptr) const = 0;
  virtual void resetOrCreateGradient(NameMappingFN fn = nullptr) const = 0;
  virtual void printMean(NameMappingFN fn = nullptr) const = 0;
  virtual void run(bool debug = false) const = 0;

 public:
  std::unique_ptr<Tensor> getParamInGraph(const std::string& name) {
    if (boost::algorithm::contains(name, ".param") && !boost::algorithm::contains(name, ".grad")) {
      auto t = new Tensor();
      t->buffer_ = memory::TensorBuffer::gTensorBuffers[name];
      t->attr_ = graph_.tensors_.at(name);
      return std::unique_ptr<Tensor>(t);
    } else {
      return nullptr;
    }
  }

  std::unique_ptr<Tensor> getGradInGraph(const std::string& name) {
    if (boost::algorithm::contains(name, ".grad")) {
      auto t = new Tensor();
      t->attr_ = graph_.tensors_.at(name);
      t->buffer_ = memory::TensorBuffer::createOrResizeBuffer<float>(t->attr_->name_, t->attr_->dims_);
      return std::unique_ptr<Tensor>(t);
    } else {
      return nullptr;
    }
  }

 protected:
  const graph::Graph& graph_;
};

class NaiveEngine : public Engine {
 public:
  NaiveEngine(const graph::Graph& graph) : Engine{graph} {}

  void randomize(Engine::NameMappingFN fn = nullptr) const override;
  void resetOrCreateGradient(NameMappingFN fn = nullptr) const override;
  void run(bool debug = false) const override;
  void printMean(NameMappingFN fn = nullptr) const override;

 private:
  void accessTensor(NameMappingFN fn, std::function<void(Tensor&)> tensorFN) const;
};
}
}