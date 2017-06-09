#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void FCOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto X = cast<Matrix>(inputs[0]);
  auto W = cast<Matrix>(inputs[1]);
  auto O = cast<Matrix>(outputs[0]);
  O = X * W;
  if (inputs.size() == 3) {
    auto B = eigen::cast<eigen::Vector>(inputs[2]);
    O.rowwise() += B.transpose();
  }
}
static void FCOpShape(const SmallVec<graph::TensorAttrPtr> &inputs, const SmallVec<graph::TensorAttrPtr> &outputs) {
  auto X = inputs[0];
  auto W = inputs[1];
  outputs[0]->dims_ = {X->dims_[0], W->dims_[1]};
}

static void FCGradOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                         const Map<std::string, Any> &attrs) {
  auto X = cast<Matrix>(inputs[0]);
  auto W = cast<Matrix>(inputs[1]);
  auto GO = cast<Matrix>(inputs[2]);
  auto GW = cast<Matrix>(outputs[0]);
  // backward mul
  GW = X.transpose() * GO;
  if (outputs[1].attr_ != nullptr) {
    auto GX = cast<Matrix>(outputs[1]);
    GX = GO * W.transpose();
  }
  if (outputs[2].attr_ != nullptr) {
    auto GB = eigen::cast<Vector>(outputs[2]);
    GB = GO.colwise().sum();
  }
}

static void FCGradShapeImpl(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  auto X = inputs[0];
  auto W = inputs[1];
  outputs[0]->dims_ = W->dims_;
  if (outputs[1]) {
    outputs[1]->dims_ = X->dims_;
  }
  if (outputs[2]) {
    outputs[2]->dims_ = {W->dims_[1], 1};
  }
}

static SmallVec<graph::Op> GetFCGradImpl(const SmallVec<TensorAttrPtr> &I, const SmallVec<TensorAttrPtr> &O,
                                         const SmallVec<TensorAttrPtr> &OG, const SmallVec<TensorAttrPtr> &IG) {
  graph::Op op;
  op.type_ = "fc_grad";
  op.inputs_ = {I[0], I[1], OG[0]};
  op.outputs_ = {IG[1], IG[0], IG[2]};
  return {op};
}

static InitFunction init([] {
  {
    graph::OpMeta meta;
    meta.type_ = "fc";
    meta.kernels[graph::kDEVICE_CPU] = FCOpImpl;
    meta.shapeInferer_ = FCOpShape;
    meta.grad = GetFCGradImpl;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    graph::OpMeta meta;
    meta.type_ = "fc_grad";
    meta.kernels[graph::kDEVICE_CPU] = FCGradOpImpl;
    meta.shapeInferer_ = FCGradShapeImpl;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}