//
// Created by baidu on 2017/6/5.
//

#include "engine/Engine.h"
#include "misc/InitFunction.h"
namespace nnet {
namespace engine {
static void sigmoidOpImpl(const SmallVec<Tensor> &inputs,
                          SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto a = castToEigenArray1D(inputs[0]);
  auto o = castToEigenArray1DMutable(outputs[0]);
  o = Eigen::tanh(a);
}

static void sigmoidOpGrad(const SmallVec<Tensor>& inputs,
                              SmallVec<Tensor>& outputs,
                              const Map<std::string, Any> &attrs) {
  auto O = castToEigenArray1D(inputs[0]);
  auto OG = castToEigenArray1D(inputs[1]);
  auto IG = castToEigenArray1DMutable(outputs[0]);
  IG = OG;
  IG *= 1 - O * O;
}

static void sigmoidOpGradShape(const SmallVec<graph::TensorAttrPtr> &inputs,
                               const SmallVec<graph::TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static void sigmoidShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs,
                             const SmallVec<graph::TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static SmallVec<graph::Op> GetSigmoidGradImpl(
    const SmallVec<graph::TensorAttrPtr> &I,
    const SmallVec<graph::TensorAttrPtr> &O,
    const SmallVec<graph::TensorAttrPtr> &OG,
    const SmallVec<graph::TensorAttrPtr> &IG) {
  graph::Op op;
  op.type_ = "sigmoid_grad";
  op.inputs_ = {O[0], OG[0]};
  op.outputs_ = {IG[0]};
  return {op};
}


static util::InitFunction __init__([] {
  {
    graph::OpMeta meta;
    meta.type_ = "sigmoid";
    meta.kernels[graph::kDEVICE_CPU] = sigmoidOpImpl;
    meta.shapeInferer_ = sigmoidShapeImpl;
    meta.grad = GetSigmoidGradImpl;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    graph::OpMeta meta;
    meta.type_ = "sigmoid_grad";
    meta.kernels[graph::kDEVICE_CPU] = sigmoidOpGrad;
    meta.shapeInferer_ = sigmoidOpGradShape;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}