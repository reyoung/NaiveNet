#include "Engine.h"
#include "Register.h"

namespace nnet {
namespace engine {

static void MeanOpImpl(const SmallVec<Tensor> &inputs,
                       SmallVec<Tensor> &outputs,
                       const Map<std::string, Any> &attrs) {
  auto i = castToEigenArray1D(inputs[0]);
  *(float*)(outputs[0].buffer_->get()) = i.mean();
}

static void MeanGradImpl(const SmallVec<Tensor>& inputs,
                         SmallVec<Tensor>& outputs,
                         const Map<std::string, Any>& attrs) {
  auto og = castToEigenArray1D(inputs[1]); // output grad;
  auto ig = castToEigenArray1DMutable(outputs[0]); // input grad;
  float g = *og.data();
  ig = g / ig.size();
}

static void MeanGradShapeImpl(const SmallVec<graph::TensorAttr *> &inputs,
                              const SmallVec<graph::TensorAttr *> &outputs) {
  CHECK_EQ(inputs.size(), 2);
  outputs[0]->dims_ = inputs[0]->dims_;
  CHECK_EQ(details::product(inputs[1]->dims_), 1);
}

static void MeanShapeImpl(const SmallVec<graph::TensorAttr *> &inputs,
                          const SmallVec<graph::TensorAttr *> &outputs) {
  outputs[0]->dims_ = {1, 1};
}

static SmallVec<graph::Op> GetMeanGradOp(
    const SmallVec<graph::TensorAttr*>& I,
    const SmallVec<graph::TensorAttr*>& O,
    const SmallVec<graph::TensorAttr*>& OG,
    const SmallVec<graph::TensorAttr*>& IG) {
  graph::Op op;
  op.type_ = "mean_grad";
  op.inputs_ = {I[0], OG[0]};
  op.outputs_ = {IG[0]};

  return {op};
}

static util::InitFunction init([]{
  {
    graph::OpMeta meta;
    meta.type_ = "mean";
    meta.kernels[graph::kDEVICE_CPU] = MeanOpImpl;
    meta.shapeInferer_ = MeanShapeImpl;
    meta.grad = GetMeanGradOp;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    graph::OpMeta gradMeta;
    gradMeta.type_ = "mean_grad";
    gradMeta.kernels[graph::kDEVICE_CPU] = MeanGradImpl;
    gradMeta.shapeInferer_ = MeanGradShapeImpl;
    graph::OpMeta::gAllOpMeta_[gradMeta.type_] = gradMeta;
  }
});

}
}