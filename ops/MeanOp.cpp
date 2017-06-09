#include "misc/CastEigen.h"
#include "engine/Engine.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace engine {

static void MeanOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto i = eigen::cast<eigen::Vector >(inputs[0]).array();
  *(float *)(outputs[0].buffer_->get()) = i.mean();
}

static void MeanGradImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                         const Map<std::string, Any> &attrs) {
  auto og = eigen::cast<eigen::Vector >(inputs[1]).array();   // output grad;
  auto ig = eigen::cast<eigen::Vector >(outputs[0]).array();  // input grad;
  float g = *og.data();
  ig = g / ig.size();
}

static void MeanGradShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs,
                              const SmallVec<graph::TensorAttrPtr> &outputs) {
  CHECK_EQ(inputs.size(), 2);
  outputs[0]->dims_ = inputs[0]->dims_;
  CHECK_EQ(details::product(inputs[1]->dims_), 1);
}

static void MeanShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs, const SmallVec<graph::TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = {1, 1};
}

static SmallVec<graph::Op> GetMeanGradOp(const SmallVec<graph::TensorAttrPtr> &I,
                                         const SmallVec<graph::TensorAttrPtr> &O,
                                         const SmallVec<graph::TensorAttrPtr> &OG,
                                         const SmallVec<graph::TensorAttrPtr> &IG) {
  graph::Op op;
  op.type_ = "mean_grad";
  op.inputs_ = {I[0], OG[0]};
  op.outputs_ = {IG[0]};

  return {op};
}

static util::InitFunction init([] {
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