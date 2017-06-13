#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include "graph/ComputationGraph.h"
namespace nnet {

// some typedefs for eigen matrix, and inplace casting from tensor to eigen matrix.
namespace eigen {
// Naive Net is using row major matrix, because it is easy for user
// create input data.
using Vector = Eigen::Matrix<float, Eigen::Dynamic, 1>;  // for vector, row major and col major are same.
using IVector = Eigen::Matrix<int, Eigen::Dynamic, 1>;   // Int Vector.
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Tensor = graph::Variable;

// cast for matrix like
template <typename T>
inline Eigen::Map<std::enable_if_t<!T::IsVectorAtCompileTime, T>> cast(const Tensor& t) {
  CHECK_EQ(t.attr_->dims_.size(), 2);
  return Eigen::Map<T>(reinterpret_cast<typename T::value_type*>(t.buffer_->get()), t.attr_->dims_[0],
                       t.attr_->dims_[1]);
}

// cast for vector like
template <typename T>
inline Eigen::Map<std::enable_if_t<T::IsVectorAtCompileTime, T>> cast(const Tensor& t) {
  return Eigen::Map<T>(reinterpret_cast<typename T::value_type*>(t.buffer_->get()), details::product(t.attr_->dims_));
};

}  // namespace eigen
}  // namespace nnet