#include "Engine.h"
#include "Register.h"
namespace nnet {
namespace engine {
static void addBiasOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                      const Map<std::string, Any> &attrs) {
  auto a = castToEigenMat(inputs[0]);
  auto b = castToEigenVec(inputs[1]);
  auto o = castToEigenMatMutable(outputs[0]);
  o = a;
  o.rowwise() += b.transpose();
}

static void addBiasShapeInfererImpl(const SmallVec<graph::TensorAttr *> &inputs,
                                const SmallVec<graph::TensorAttr *> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
  CHECK_EQ(inputs[0]->dims_, outputs[0]->dims_);
}

static util::InitFunction __init__([] {
  graph::OpMeta meta;
  meta.type_ = "add_bias";
  meta.kernels[graph::kDEVICE_CPU] = addBiasOpImpl;
  meta.shapeInferer_ = addBiasShapeInfererImpl;
  graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
});
}
}