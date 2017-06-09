#include "engine/Engine.h"
#include "misc/InitFunction.h"
#include "misc/CastEigen.h"

namespace nnet {
namespace engine {
static void lookupTable(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto WORD = eigen::cast<eigen::IVector>(inputs[0]).array();
  auto WEIGHT = eigen::cast<eigen::Matrix>(inputs[1]).array();
  auto O = eigen::cast<eigen::Matrix>(outputs[0]).array();

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

static void lookupTableGrad(const SmallVec<Tensor>& inputs, SmallVec<Tensor>& outputs, const Map<std::string, Any> &attrs) {
  auto WORD = eigen::cast<eigen::Matrix >(inputs[0]).array();
  auto OG = eigen::cast<eigen::Matrix>(inputs[1]).array();

  auto WG = eigen::cast<eigen::Matrix>(outputs[0]).array();
  WG = OG;
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