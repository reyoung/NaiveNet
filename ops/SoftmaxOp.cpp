#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void softmaxOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto X = cast<Matrix>(inputs[0]).array();
  auto P = cast<Matrix>(outputs[0]).array();

  P = X.exp();
  P.colwise() /= P.rowwise().sum();
}

static void softmaxShapeImpl(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static void softmaxGradImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                            const Map<std::string, Any> &attrs) {
  auto Y = cast<Matrix>(inputs[0]);
  auto DY = cast<Matrix>(inputs[1]);
  auto DX = cast<Matrix>(outputs[0]);
  DX.array() = DY.array();
  for (size_t i=0; i<inputs[0].attr_->dims_[0]; ++i) {
    float dot = Y.row(i).dot(DY.row(i));
    DX.row(i).array() -= dot;
  }
  DX.array() *= Y.array();
}
static void softmaxGradShapeImpl(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  auto P = inputs[0];
  auto OG = inputs[1];
  auto IG = outputs[0];
  IG->dims_ = P->dims_;
  CHECK_EQ(OG->dims_, P->dims_);
}

static SmallVec<Op> GetSoftmaxGradOp(const SmallVec<TensorAttrPtr> &I, const SmallVec<TensorAttrPtr> &O,
                                     const SmallVec<TensorAttrPtr> &OG, const SmallVec<TensorAttrPtr> &IG) {
  Op op;
  op.type_ = "softmax_grad";
  op.inputs_ = {O[0], OG[0]};
  op.outputs_ = {IG[0]};

  return {op};
}

static util::InitFunction init([] {
  {
    OpMeta meta;
    meta.type_ = "softmax";
    meta.kernels[kDEVICE_CPU] = softmaxOpImpl;
    meta.shapeInferer_ = softmaxShapeImpl;
    meta.grad_ = GetSoftmaxGradOp;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    OpMeta meta;
    meta.type_ = "softmax_grad";
    meta.kernels[kDEVICE_CPU] = softmaxGradImpl;
    meta.shapeInferer_ = softmaxGradShapeImpl;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}