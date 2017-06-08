#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {
static void shapeInferer(Graph& g, const Map<std::string, Any>& ignored) {
  for (auto& op: g.ops_) {
    auto& opMeta = OpMeta::gAllOpMeta_[op.type_];
    opMeta.shapeInferer_(op.inputs_, op.outputs_);
  }
}
static util::InitFunction init([]{
  compilers().insert({"inferenceShape", shapeInferer});
});

}
}