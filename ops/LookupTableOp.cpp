#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {
static void lookupTable(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto WORD = cast<IVector>(inputs[0]).array();
  auto WEIGHT = cast<Matrix>(inputs[1]).array();
  auto O = cast<Matrix>(outputs[0]).array();

  for (size_t i = 0; i < WORD.rows(); ++i) {
    auto offset = WORD.data()[i];
    CHECK_LT(offset, WEIGHT.rows());
    O.row(i) = WEIGHT.row(offset);
  }
}

static void fwdShape(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  auto WORD = inputs[0];
  auto WEIGHT = inputs[1];
  auto O = outputs[0];
  O->dims_ = {details::product(WORD->dims_), WEIGHT->dims_[1]};
}

static void lookupTableGrad(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs,
                            const Map<std::string, Any> &attrs) {
  auto OG = cast<Matrix>(inputs[0]).array();
  auto WG = cast<Matrix>(outputs[0]).array();
  WG = OG;

  int * JA = (int*) outputs[2].buffer_->get();
  int * IA = (int*) outputs[1].buffer_->get();
  *IA = 0;
  ++IA;
  for (size_t i=0; i < inputs[0].attr_->dims_[0]; ++i, ++IA) {
    *IA = (int)((i+1) * outputs[0].attr_->dims_[1]);
    for (int j=0; j<outputs[0].attr_->dims_[1]; ++j) {
      JA[j] = j;
    }
    JA += outputs[0].attr_->dims_[1];
  }
}

static void lookupTableShape(const SmallVec<TensorAttrPtr> &inputs, const SmallVec<TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_; // same dim float
  outputs[1]->dims_ = {inputs[0]->dims_[0] + 1, 1UL};
  outputs[2]->dims_ = {inputs[0]->dims_[0] * inputs[0]->dims_[1], 1UL};
}

static util::InitFunction init([] {
  {
    OpMeta meta;
    meta.type_ = "lookup_table";
    meta.kernels[kDEVICE_CPU] = lookupTable;
    meta.shapeInferer_ = fwdShape;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
  {
    OpMeta meta;
    meta.type_ = "lookup_table_grad";
    meta.kernels[kDEVICE_CPU] = lookupTableGrad;
    meta.shapeInferer_ = lookupTableShape;
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}