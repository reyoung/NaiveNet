//
// Created by baidu on 2017/6/5.
//

#include "Engine.h"
#include "Register.h"
namespace nnet {
namespace engine {
static void sigmoidOpImpl(const SmallVec<Tensor> &inputs,
                          SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto a = castToEigenArray1D(inputs[0]);
  auto o = castToEigenArray1DMutable(outputs[0]);
  o = Eigen::tanh(a);
}

static void sigmoidShapeImpl(const SmallVec<graph::TensorAttr *> &inputs,
                             const SmallVec<graph::TensorAttr *> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static util::InitFunction __init__([] {
  graph::OpMeta meta;
  meta.type_ = "sigmoid";
  meta.kernels[graph::kDEVICE_CPU] = sigmoidOpImpl;
  meta.shapeInferer_ = sigmoidShapeImpl;
  graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  return Error();
});
}
}