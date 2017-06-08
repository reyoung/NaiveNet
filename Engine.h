#pragma once
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include "ComputationGraph.h"

namespace nnet {
namespace engine {
using graph::Tensor;

inline Eigen::Map<const Eigen::VectorXf> castToEigenVec(const Tensor& tensor) {
  return Eigen::Map<const Eigen::VectorXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()), tensor.attr_->dims_[0]);
}

inline Eigen::Map<Eigen::VectorXf> castToEigenVecMutable(const Tensor& tensor) {
  return Eigen::Map<Eigen::VectorXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()), tensor.attr_->dims_[0]);
}

inline Eigen::Map<const Eigen::MatrixXf> castToEigenMat(const Tensor& tensor) {
  CHECK_EQ(tensor.attr_->dims_.size(), 2UL);
  return Eigen::Map<const Eigen::MatrixXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()), tensor.attr_->dims_[0],
      tensor.attr_->dims_[1]);
}

inline Eigen::Map<const Eigen::ArrayXi> castToEigenIArray1D(
    const Tensor& tensor) {
  return Eigen::Map<const Eigen::ArrayXi>((int*)tensor.buffer_->get(),
                                          tensor.attr_->dims_[0]);
}

inline Eigen::Map<const Eigen::MatrixXi> castToEigenIMat(const Tensor& tensor) {
  CHECK_EQ(tensor.attr_->dims_.size(), 2UL);
  return Eigen::Map<const Eigen::MatrixXi>(
      reinterpret_cast<int*>(tensor.buffer_->get()), tensor.attr_->dims_[0],
      tensor.attr_->dims_[1]);
}

inline Eigen::Map<Eigen::MatrixXf> castToEigenMatMutable(Tensor& tensor) {
  CHECK_EQ(tensor.attr_->dims_.size(), 2UL);
  return Eigen::Map<Eigen::MatrixXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()), tensor.attr_->dims_[0],
      tensor.attr_->dims_[1]);
}

inline Eigen::Map<Eigen::ArrayXXf> castToEigenArray2DMutable(
    const Tensor& tensor) {
  return Eigen::Map<Eigen::ArrayXXf>((float*)tensor.buffer_->get(),
                                     tensor.attr_->dims_[0],
                                     tensor.attr_->dims_[1]);
}

inline Eigen::Map<const Eigen::ArrayXf> castToEigenArray1D(
    const Tensor& tensor) {
  return Eigen::Map<const Eigen::ArrayXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()),
      details::product(tensor.attr_->dims_));
}

inline Eigen::Map<const Eigen::ArrayXXf> castToEigenArray2D(
    const Tensor& tensor) {
  return Eigen::Map<const Eigen::ArrayXXf>((float*)tensor.buffer_->get(),
                                           tensor.attr_->dims_[0],
                                           tensor.attr_->dims_[1]);
}

inline Eigen::Map<Eigen::ArrayXf> castToEigenArray1DMutable(
    const Tensor& tensor) {
  return Eigen::Map<Eigen::ArrayXf>(
      reinterpret_cast<float*>(tensor.buffer_->get()),
      details::product(tensor.attr_->dims_));
}

class Engine {
 public:
  using NameMappingFN =
      std::function<std::unique_ptr<Tensor>(const std::string& name)>;
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