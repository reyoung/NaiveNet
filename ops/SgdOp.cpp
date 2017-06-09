#include "engine/Engine.h"
#include "misc/InitFunction.h"
#include "misc/CastEigen.h"

namespace nnet {
namespace engine {

static void SgdOpImpl(const SmallVec<Tensor> &inputs, SmallVec<Tensor> &outputs, const Map<std::string, Any> &attrs) {
  auto V = eigen::cast<eigen::Vector>(inputs[0]).array();
  auto G = eigen::cast<eigen::Vector>(inputs[1]).array();
  auto Target = eigen::cast<eigen::Vector>(outputs[0]).array();
  float learning_rate = any_cast<float>(attrs.at("learning_rate"));
  Target = V - learning_rate * G;
}

static void SgdShapeImpl(const SmallVec<graph::TensorAttrPtr> &inputs, const SmallVec<graph::TensorAttrPtr> &outputs) {
  outputs[0]->dims_ = inputs[0]->dims_;
}

static util::InitFunction init([] {
  {
    graph::OpMeta meta;
    meta.type_ = "sgd";
    meta.kernels[graph::kDEVICE_CPU] = SgdOpImpl;
    meta.shapeInferer_ = SgdShapeImpl;
    meta.attrMeta_.push_back(graph::AttributeMeta::create<float>("learning_rate", "LR for sgd"));
    auto &attrMeta = meta.attrMeta_.back();
    attrMeta->constraints<float>().defaultValue(0.0001);
    graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  }
});
}
}