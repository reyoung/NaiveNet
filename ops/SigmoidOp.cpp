#include "EigenOp-inl.h"
namespace nnet {
namespace eigen_ops {
static void sigmoidOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto a = cast<Vector>(inputs[0]).array();
  auto o = cast<Vector>((outputs[0])).array();
  o = tanh(a);
}

static void sigmoidOpGrad(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto O = cast<Vector>(inputs[0]).array();
  auto OG = cast<Vector>(inputs[1]).array();
  auto IG = cast<Vector>(outputs[0]).array();
  IG = OG;
  IG *= 1 - O * O;
}

static void sigmoidOpGradShape(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static void sigmoidShapeImpl(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static SmallVec<Op> GetSigmoidGradImpl(const SmallVec<TensorAttrPtr> &I, const SmallVec<TensorAttrPtr> &O,
                                       const SmallVec<TensorAttrPtr> &OG, const SmallVec<TensorAttrPtr> &IG) {
  Op op;
  op.type_ = "sigmoid_grad";
  op.inputs_ = {O[0], OG[0]};
  op.outputs_ = {IG[0]};
  return {op};
}

static util::InitFunction __init__([] {
  {
    OpMeta meta;
    meta.type_ = "sigmoid";
    meta.kernels[kDEVICE_CPU] = sigmoidOpImpl;
    meta.shapeInferer_ = sigmoidShapeImpl;
    meta.grad_ = GetSigmoidGradImpl;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    OpMeta meta;
    meta.type_ = "sigmoid_grad";
    meta.kernels[kDEVICE_CPU] = sigmoidOpGrad;
    meta.shapeInferer_ = sigmoidOpGradShape;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}