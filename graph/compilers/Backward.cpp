#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {
static void backward(Graph& g, const Map<std::string, Any>& attrs) {
  CHECK_GT(g.ops_.size(), 0);
  auto lossName = any_cast<std::string>(attrs.at("loss_name"));
  auto& lossVarAttr = g.variables_.at(lossName);
  CHECK_EQ(details::product(lossVarAttr->dims_), 1UL);
  CHECK_EQ(lossVarAttr->type_, kFLOAT32);
  lossVarAttr->specialResetFunction_ = [](Variable t) { *(float*)(t.buffer_->get()) = 1.0; };
  size_t backwardPoint;
  {
    auto it = attrs.find("backward_point");
    if (it != attrs.end()) {
      backwardPoint = any_cast<size_t>(it->second);
    } else {
      backwardPoint = g.ops_.size() - 1;
    }
  }
  for (; backwardPoint != -1UL; --backwardPoint) {  // backward each op
    auto& op = g.ops_[backwardPoint];
    auto& opMeta = graph::OpMeta::gAllOpMeta_[op.type_];

    SmallVec<VariableAttrPtr> I = op.inputs_;
    SmallVec<VariableAttrPtr> O = op.outputs_;
    SmallVec<VariableAttrPtr> OG;
    SmallVec<VariableAttrPtr> IG;
    opMeta.gradVars_(I, O, &OG, &IG);

    for (auto& og : OG) {
      if (!og) continue;
      og = g.createOrResizeVar<true>(og);
    }
    for (auto& ig : IG) {
      if (!ig) continue;
      ig = g.createOrResizeVar<true>(ig);
    }

    auto ops = opMeta.grad_(I, O, OG, IG);
    for (auto& o : ops) {
      g.ops_.push_back(o);
    }
  }
}

static util::InitFunction init([] { compilers().insert({"backward", backward}); });
}
}