#include "graph/ComputationGraph.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace graph {

static void requestOrResizeBufferImpl(Graph& g,
                                      const Map<std::string, Any>& ignored) {
  g.buffers_.clear();
  for (auto& attrPair : g.tensors_) {
    auto& attrPtr = attrPair.second;
    memory::TensorBufferPtr buf;
    switch (attrPtr->type_) {
      case kTENSOR_FLOAT32:
        buf = memory::TensorBuffer::createOrResizeBuffer<float>(attrPtr->name_,
                                                                attrPtr->dims_);
        break;
      case kTENSOR_INT32:
        buf = memory::TensorBuffer::createOrResizeBuffer<int>(attrPtr->name_,
                                                              attrPtr->dims_);
        break;
    }
    g.buffers_.insert({attrPair.first, buf});
  }
}

static util::InitFunction init([] {
  compilers().insert({"requestResource", requestOrResizeBufferImpl});
});
}
}