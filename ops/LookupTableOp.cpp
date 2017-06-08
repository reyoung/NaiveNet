#include "engine/Engine.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace engine {
static void lookupTable(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto WORD = castToEigenIArray1D(inputs[0]);
  auto WEIGHT = castToEigenMat(inputs[1]).array();
  auto O = castToEigenArray2DMutable(outputs[0]);

  for (size_t i=0; i<WORD.rows(); ++i) {
    auto offset = WORD.data()[i];
    CHECK_LT(offset, WEIGHT.rows());
    O.row(i) = WEIGHT.row(offset);
  }
}

static void fwdShape(const SmallVec<graph::TensorAttrPtr> &inputs, const SmallVec<graph::TensorAttrPtr> &outputs) {
  auto WORD = inputs[0];
  auto WEIGHT = inputs[1];
  auto O = outputs[0];
  O->dims_ = {details::product(WORD->dims_), WEIGHT->dims_[1]};
}

static util::InitFunction init([]{
  {
    graph::OpMeta meta;
    meta.type_ = "lookup_table";
    meta.kernels[graph::kDEVICE_CPU] = lookupTable;
    meta.shapeInferer_ = fwdShape;
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});

}
}