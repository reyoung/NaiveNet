#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void MeanOpImpl(const SmallVec<Variable> &inputs, SmallVec<Variable> &outputs,
                       const Map<std::string, Any> &attrs) {
  auto i = cast<Vector>(inputs[0]).array();
  *(float *)(outputs[0].buffer_->get()) = i.mean();
}

static void MeanGradImpl(const SmallVec<Variable> &inputs, SmallVec<Variable> &outputs,
                         const Map<std::string, Any> &attrs) {
  auto og = cast<Vector>(inputs[1]).array();   // output grad_;
  auto ig = cast<Vector>(outputs[0]).array();  // input grad_;
  float g = *og.data() / ig.size();
  ig = g;
}

static void MeanGradShapeImpl(const SmallVec<VariableAttrPtr> &inputs, const SmallVec<VariableAttrPtr> &outputs) {
  CHECK_EQ(inputs.size(), 2);
  outputs[0]->dims_ = inputs[0]->dims_;
  CHECK_EQ(details::product(inputs[1]->dims_), 1);
}

static void MeanShapeImpl(const SmallVec<VariableAttrPtr> &inputs, const SmallVec<VariableAttrPtr> &outputs) {
  outputs[0]->dims_ = {1, 1};
}

static SmallVec<Op> GetMeanGradOp(const SmallVec<VariableAttrPtr> &I, const SmallVec<VariableAttrPtr> &O,
                                  const SmallVec<VariableAttrPtr> &OG, const SmallVec<VariableAttrPtr> &IG) {
  Op op;
  op.type_ = "mean_grad";
  op.inputs_ = {I[0], OG[0]};
  op.outputs_ = {IG[0]};

  return {op};
}

static util::InitFunction init([] {
  {
    OpMeta meta;
    meta.type_ = "mean";
    meta.kernels[kDEVICE_CPU] = MeanOpImpl;
    meta.shapeInferer_ = MeanShapeImpl;
    meta.grad_ = GetMeanGradOp;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    OpMeta gradMeta;
    gradMeta.type_ = "mean_grad";
    gradMeta.kernels[kDEVICE_CPU] = MeanGradImpl;
    gradMeta.shapeInferer_ = MeanGradShapeImpl;

    OpMeta::gAllOpMeta_[gradMeta.type_] = gradMeta;
  }
});
}
}