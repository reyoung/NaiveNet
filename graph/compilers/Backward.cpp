#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {

static TensorAttrPtr transformGradient(const TensorAttrPtr& ptr) {
  if (ptr->needBackward_) {
    auto retv = std::make_shared<TensorAttr>();
    *retv = *ptr;
    retv->name_ = ptr->name_ + ".grad";
    return retv;
  } else {
    return nullptr;
  }
}

static void backward(Graph& g, const Map<std::string, Any>& attrs) {
  CHECK_GT(g.ops_.size(), 0);
  auto lossName = any_cast<std::string>(attrs.at("loss_name"));
  auto& lossTensorAttr = g.tensors_.at(lossName);
  CHECK_EQ(details::product(lossTensorAttr->dims_), 1UL);
  CHECK_EQ(lossTensorAttr->type_, kTENSOR_FLOAT32);
  lossTensorAttr->specialResetFunction_ = [](Tensor t) { *(float*)(t.buffer_->get()) = 1.0; };
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

    SmallVec<TensorAttrPtr> I = op.inputs_;
    SmallVec<TensorAttrPtr> O = op.outputs_;
    SmallVec<TensorAttrPtr> OG(O.size());
    SmallVec<TensorAttrPtr> IG(I.size());
    std::transform(I.begin(), I.end(), IG.begin(), transformGradient);
    std::transform(O.begin(), O.end(), OG.begin(), transformGradient);

    for (auto& og : OG) {
      if (!og) continue;
      g.tensors_.insert({og->name_, og});
    }
    for (auto& ig : IG) {
      if (!ig) continue;
      g.tensors_.insert({ig->name_, ig});
    }
    auto ops = opMeta.grad(I, O, OG, IG);
    for (auto& o : ops) {
      g.ops_.push_back(o);
    }
  }
}

static util::InitFunction init([] { compilers().insert({"backward", backward}); });
}
}