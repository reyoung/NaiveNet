#pragma once
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "graph/ComputationGraph.h"
#include "memory/Workspace.h"

namespace nnet {
namespace engine {
using graph::Tensor;

class Engine {
 public:
  using NameMappingFN = std::function<std::unique_ptr<Tensor>(const std::string& name)>;
  Engine(memory::Workspace& w, const graph::Graph& graph) : graph_{graph}, workspace_(w) {}
  virtual ~Engine() {}

  virtual void randomize(NameMappingFN fn = nullptr) const = 0;
  virtual void resetOrCreateGradient(NameMappingFN fn = nullptr) const = 0;
  virtual void printMean(NameMappingFN fn = nullptr) const = 0;
  virtual void run(bool debug = false) const = 0;

 public:
  std::unique_ptr<Tensor> getParamInGraph(const std::string& name) {
    if (boost::algorithm::contains(name, ".param") && !boost::algorithm::contains(name, ".grad")) {
      auto t = new Tensor();
      *t = workspace_.getTensor(graph_.tensors_.at(name));
      return std::unique_ptr<Tensor>(t);
    } else {
      return nullptr;
    }
  }

  std::unique_ptr<Tensor> getGradInGraph(const std::string& name) {
    if (boost::algorithm::contains(name, ".grad")) {
      auto t = new Tensor();
      *t = workspace_.getTensor(graph_.tensors_.at(name));
      return std::unique_ptr<Tensor>(t);
    } else {
      return nullptr;
    }
  }

 protected:
  const graph::Graph& graph_;
  memory::Workspace& workspace_;
};

class NaiveEngine : public Engine {
 public:
  NaiveEngine(memory::Workspace& w, const graph::Graph& graph) : Engine(w, graph) {}

  void randomize(Engine::NameMappingFN fn = nullptr) const override;
  void resetOrCreateGradient(NameMappingFN fn = nullptr) const override;
  void run(bool debug = false) const override;
  void printMean(NameMappingFN fn = nullptr) const override;

 private:
  void accessTensor(NameMappingFN fn, std::function<void(Tensor&)> tensorFN) const;
};
}
}