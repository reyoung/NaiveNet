#include "graph/ComputationGraph.h"
#include "memory/Workspace.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {
static void reqRes(Graph &g, const Map<std::string, Any> &attrs) {
  auto w = any_cast<memory::Workspace *>(attrs.at("workspace"));
  for (auto &t : g.variables_) {
    (*w)(t.second);
  }
}
static util::InitFunction init([] { compilers().insert({"requestResource", reqRes}); });
}
}