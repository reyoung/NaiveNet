//
// Created by baidu on 2017/6/5.
//

#include "Engine.h"
#include "Register.h"
namespace nnet {
namespace engine {
static void mulOpImpl(const SmallVec<engine::Tensor> &inputs,
                      SmallVec<engine::Tensor> &outputs,
                      const Map<std::string, Any> &attrs) {
  auto a = castToEigenMat(inputs[0]);
  auto b = castToEigenMat(inputs[1]);
  auto o = castToEigenMatMutable(outputs[0]);
  float scale = any_cast<float>(attrs.at("scale"));
  o = a * b;
  o *= scale;
}

static void mulOpShapeInferer(const SmallVec<graph::TensorAttr *> &inputs,
                              const SmallVec<graph::TensorAttr *> &outputs) {
  outputs[0]->dims_ = {inputs[0]->dims_[0], inputs[1]->dims_[1]};
}

static util::InitFunction __init__([] {
  graph::OpMeta meta;
  meta.type_ = "mul";
  meta.kernels[graph::kDEVICE_CPU] = mulOpImpl;
  meta.shapeInferer_ = mulOpShapeInferer;
  meta.attrMeta_.push_back(
      graph::AttributeMeta::create<float>("scale", "scale apply to mul op"));
  auto &scaleMeta = meta.attrMeta_.back();
  scaleMeta->constraints<float>().defaultValue(1.0);
  graph::OpMeta::gAllOpMeta_[meta.type_] = meta;
  return Error();
});
}
}