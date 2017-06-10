#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void XEOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto p = (float *)(inputs[0].buffer_->get());
  auto l = (int *)(inputs[1].buffer_->get());
  auto loss = (float *)(outputs[0].buffer_->get());

  auto batchSize = inputs[0].attr_->dims_[0];
  auto featureSize = inputs[0].attr_->dims_[1];

  for (decltype(batchSize) i = 0; i < batchSize; ++i) {
    auto label = l[i];
    CHECK_LT(label, featureSize);
    loss[i] = -std::log(p[featureSize * i + l[i]]);
  }
}

static void XEShapeImpl(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  CHECK_EQ(inputs[0]->dims_[0], inputs[1]->dims_[0]);
  CHECK_EQ(inputs[1]->type_, graph::kTENSOR_INT32);
  outputs[0]->dims_ = {inputs[0]->dims_[0], 1};
}

static void XEOpGradImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                         const Map<std::string, Any> &attrs) {
  size_t numSamples = inputs[0].attr_->dims_[0];
  size_t dim = inputs[0].attr_->dims_[1];
  float *out = (float *)inputs[0].buffer_->get();
  float *grad = (float *)outputs[0].buffer_->get();
  int *lbl = (int *)inputs[1].buffer_->get();
  for (size_t i = 0; i < numSamples; ++i, out += dim, grad += dim) {
    grad[lbl[i]] -= 1 / out[lbl[i]];
  }
  auto GO = cast<Vector>(inputs[2]).array();  // coeff
  auto GI = cast<Matrix>(outputs[0]).array();
  GI.colwise() *= GO;
}

static void XEGradShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs,
                            const SmallVec<graph::TensorAttrPtr> &outputs) {
  CHECK_EQ(inputs.size(), 3UL);
  auto P = inputs[0];
  auto L = inputs[1];
  auto GO = inputs[2];
  auto GI = outputs[0];

  CHECK_EQ(P->dims_[0], details::product(L->dims_));
  CHECK_EQ(P->dims_[0], GO->dims_[0]);
  GI->dims_ = P->dims_;
}

static SmallVec<graph::Op> GetXeGradOp(const SmallVec<TensorAttrPtr> &I, const SmallVec<TensorAttrPtr> &O,
                                       const SmallVec<TensorAttrPtr> &OG, const SmallVec<TensorAttrPtr> &IG) {
  graph::Op op;
  op.type_ = "cross_entropy_grad";
  op.inputs_ = {I[0], I[1], OG[0]};
  op.outputs_ = {IG[0]};
  return {op};
}

static InitFunction __init__([] {
  {
    graph::OpMeta meta;
    meta.type_ = "cross_entropy";
    meta.kernels[graph::kDEVICE_CPU] = XEOpImpl;
    meta.shapeInferer_ = XEShapeImpl;
    meta.grad_ = GetXeGradOp;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    graph::OpMeta meta;
    meta.type_ = "cross_entropy_grad";
    meta.kernels[graph::kDEVICE_CPU] = XEOpGradImpl;
    meta.shapeInferer_ = XEGradShapeImpl;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}