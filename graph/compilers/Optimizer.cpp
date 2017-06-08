#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"
#include "boost/algorithm/string.hpp"

namespace nnet {
namespace graph {
static void optimizer(Graph& g, const Map<std::string, Any>& attrs) {
  for (auto tensorPtr : g.tensors_) {
    if (boost::algorithm::contains(tensorPtr.first, ".grad")
         && boost::algorithm::contains(tensorPtr.first, ".param")) {
      auto paramKey =
          boost::algorithm::replace_last_copy(tensorPtr.first, ".grad", "");
      auto it = g.tensors_.find(paramKey);
      CHECK_NE(it, g.tensors_.end());
      g.ops_.emplace_back();
      graph::Op& op = g.ops_.back();
      op.type_ = "sgd";
      op.attrs_ = attrs;
      op.inputs_ = {it->second, tensorPtr.second};
      op.outputs_ = {it->second};
      auto & meta = OpMeta::gAllOpMeta_[op.type_];
      for (auto& attrMeta : meta.attrMeta_) {
        attrMeta->constraints_->check(attrMeta->name_, &op.attrs_);
      }
    }
  }
}
static util::InitFunction init([]{
  compilers().insert({"optimizer", optimizer});
});
}
}