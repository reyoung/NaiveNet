#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void softmaxOpImpl(const SmallVec<Variable> &inputs, SmallVec<Variable> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto X = cast<Matrix>(inputs[0]).array();
  auto P = cast<Matrix>(outputs[0]).array();

  P = X.exp();
  P.colwise() /= P.rowwise().sum();
}

static void softmaxShapeImpl(const SmallVec<VariableAttrPtr> &inputs, const SmallVec<VariableAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static void softmaxGradImpl(const SmallVec<Variable> &inputs, SmallVec<Variable> &outputs,
                            const Map<std::string, Any> &attrs) {
  auto Y = cast<Matrix>(inputs[0]);
  auto DY = cast<Matrix>(inputs[1]);
  auto DX = cast<Matrix>(outputs[0]);
  DX.array() = DY.array();
  for (size_t i = 0; i < inputs[0].attr_->dims_[0]; ++i) {
    float dot = Y.row(i).dot(DY.row(i));
    DX.row(i).array() -= dot;
  }
  DX.array() *= Y.array();
}
static void softmaxGradShapeImpl(const SmallVec<VariableAttrPtr> &inputs, const SmallVec<VariableAttrPtr> &outputs) {
  auto P = inputs[0];
  auto OG = inputs[1];
  auto IG = outputs[0];
  IG->dims_ = P->dims_;
  CHECK_EQ(OG->dims_, P->dims_);
}

static SmallVec<Op> GetSoftmaxGradOp(const SmallVec<VariableAttrPtr> &I, const SmallVec<VariableAttrPtr> &O,
                                     const SmallVec<VariableAttrPtr> &OG, const SmallVec<VariableAttrPtr> &IG) {
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