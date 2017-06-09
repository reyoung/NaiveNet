#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {
static void errorRateOps(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                         const Map<std::string, Any> &attrs) {
  auto prob = cast<Matrix >(inputs[0]);
  auto lbl = cast<IVector >(inputs[1]);
  Eigen::Index  idx;
  size_t cnt = 0;
  for (decltype(prob.rows()) i = 0; i<prob.rows(); ++i) {
    prob.row(i).maxCoeff(&idx);
    auto l = lbl.data()[i];
    if (l == idx) {
      ++cnt;
    }
  }
  auto rate = cast<Vector>(outputs[0]);
  rate[0] = (float)(1.0 - (double)(cnt)/lbl.size());
}
static void errorRateShape(const SmallVec<graph::TensorAttrPtr> &i, const SmallVec<graph::TensorAttrPtr> &o) {
  CHECK_EQ(i[0]->dims_[0], details::product(i[1]->dims_));
  o[0]->dims_ = {1, 1};
}
static SmallVec<Op> NoOps(const SmallVec<TensorAttrPtr> &I, const SmallVec<TensorAttrPtr> &O,
                          const SmallVec<TensorAttrPtr> &OG, const SmallVec<TensorAttrPtr> &IG) {
  return {};
}

static InitFunction init([]{
  {
    graph::OpMeta meta;
    meta.type_ = "error_rate";
    meta.kernels[graph::kDEVICE_CPU] = errorRateOps;
    meta.shapeInferer_ = errorRateShape;
    meta.grad = NoOps;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}