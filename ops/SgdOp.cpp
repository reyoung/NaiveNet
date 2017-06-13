#include "EigenOp-inl.h"

namespace nnet {
namespace eigen_ops {

static void SgdOpImpl(const SmallVec<Variable> &inputs, SmallVec<Variable> &outputs, const Map<std::string, Any> &attrs) {
  auto V = cast<Vector>(inputs[0]).array();
  auto G = cast<Vector>(inputs[1]).array();
  auto Target = cast<Vector>(outputs[0]).array();
  float learning_rate = any_cast<float>(attrs.at("learning_rate"));
  Target = V - learning_rate * G;
}

static void SgdShapeImpl(const SmallVec<VariableAttrPtr> &inputs, const SmallVec<VariableAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static util::InitFunction init([] {
  {
    OpMeta meta;
    meta.type_ = "sgd";
    meta.kernels[kDEVICE_CPU] = SgdOpImpl;
    meta.shapeInferer_ = SgdShapeImpl;
    meta.attrMeta_.push_back(AttributeMeta::create<float>("learning_rate", "LR for sgd"));
    auto &attrMeta = meta.attrMeta_.back();
    attrMeta->constraints<float>().defaultValue(0.0001);
    OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}