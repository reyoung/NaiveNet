#include "boost/algorithm/string.hpp"
#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {
static void optimizer(Graph& g, const Map<std::string, Any>& attrs) {
  for (auto varPtr : g.variables_) {
    auto optimizer = any_cast<std::string>(attrs.at("optimizer"));
    if (boost::algorithm::contains(varPtr.first, ".grad") && boost::algorithm::contains(varPtr.first, ".param")) {
      auto paramKey = boost::algorithm::replace_last_copy(varPtr.first, ".grad", "");
      auto it = g.variables_.find(paramKey);
      CHECK_NE(it, g.variables_.end());
      g.ops_.emplace_back();
      graph::Op& op = g.ops_.back();
      op.type_ = optimizer;
      auto& meta = OpMeta::gAllOpMeta_[op.type_];

      Map<std::string, Any> opAttr;
      for (auto& attrMeta : meta.attrMeta_) {
        auto attrsIt = attrs.find(attrMeta->name_);
        if (attrsIt != attrs.end()) {
          opAttr[attrsIt->first] = attrsIt->second;
        }
      }
      op.attrs_ = opAttr;
      op.inputs_ = {it->second, varPtr.second};
      op.outputs_ = {it->second};
      for (auto& attrMeta : meta.attrMeta_) {
        attrMeta->constraints_->check(attrMeta->name_, &op.attrs_);
      }
    }
  }
}
static util::InitFunction init([] { compilers().insert({"optimizer", optimizer}); });
}
}