//
// Created by baidu on 2017/6/5.
//

#include "engine/Engine.h"
#include "misc/InitFunction.h"
#include "misc/CastEigen.h"

namespace nnet {
namespace engine {

static void softmaxOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                          const Map<std::string, Any> &attrs) {
  auto X = eigen::cast<eigen::Matrix>(inputs[0]).array();
  auto P = eigen::cast<eigen::Matrix>(outputs[0]).array();

  P = X.exp();
  P.colwise() /= P.rowwise().sum();
}

static void softmaxShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs,
                             const SmallVec<graph::TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static void softmaxGradImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                            const Map<std::string, Any> &attrs) {
  auto P = eigen::cast<eigen::Matrix>(inputs[0]);
  auto OG = eigen::cast<eigen::Matrix>(inputs[1]);
  auto IG = eigen::cast<eigen::Matrix>(outputs[0]);

  IG = OG;
  Eigen::MatrixXf mat(OG.rows(), OG.cols());
  mat = OG;
  mat.array() *= P.array();  // elemwise mul
  IG.rowwise() -= mat.colwise().sum();
  IG = IG.array() * P.array();  // elemwise mul
}
static void softmaxGradShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs,
                                 const SmallVec<graph::TensorAttrPtr> &outputs) {
  auto P = inputs[0];
  auto OG = inputs[1];
  auto IG = outputs[0];
  IG->dims_ = P->dims_;
  CHECK_EQ(OG->dims_, P->dims_);
}

static SmallVec<graph::Op> GetSoftmaxGradOp(const SmallVec<graph::TensorAttrPtr> &I,
                                            const SmallVec<graph::TensorAttrPtr> &O,
                                            const SmallVec<graph::TensorAttrPtr> &OG,
                                            const SmallVec<graph::TensorAttrPtr> &IG) {
  graph::Op op;
  op.type_ = "softmax_grad";
  op.inputs_ = {O[0], OG[0]};
  op.outputs_ = {IG[0]};

  return {op};
}

static util::InitFunction init([] {
  {
    graph::OpMeta meta;
    meta.type_ = "softmax";
    meta.kernels[graph::kDEVICE_CPU] = softmaxOpImpl;
    meta.shapeInferer_ = softmaxShapeImpl;
    meta.grad = GetSoftmaxGradOp;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    graph::OpMeta meta;
    meta.type_ = "softmax_grad";
    meta.kernels[graph::kDEVICE_CPU] = softmaxGradImpl;
    meta.shapeInferer_ = softmaxGradShapeImpl;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}