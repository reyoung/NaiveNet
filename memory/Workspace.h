#pragma once
#include "VariableBuffer.h"
#include "graph/ComputationGraph.h"

namespace nnet {
namespace memory {

class Workspace final {
public:
  Map<std::string, std::shared_ptr<VariableBuffer>> varBuffers_;

  Workspace() = default;
  Workspace(const Workspace& o) = delete;

  std::shared_ptr<VariableBuffer> createBuffer(const std::string& name, size_t size, Device dev) {
    if (dev == kDEVICE_CPU) {
      auto buf = std::make_shared<CpuVariableBuffer>(size);
      varBuffers_.insert({name, buf});
      return buf;
    } else {
      LOG(FATAL) << "Not implemented";
      return nullptr;
    }
  }

  std::shared_ptr<VariableBuffer> operator()(const graph::VariableAttrPtr & attr) {
    size_t sz = details::product(attr->dims_);
    static_assert(sizeof(float) == sizeof(int), "");
    sz *= sizeof(float);
    return createOrResizeBuffer(attr->name_, sz, kDEVICE_CPU);
  }

  graph::Variable getVar(const graph::VariableAttrPtr &attr) {
    return {attr, this->operator()(attr)};
  }

  std::shared_ptr<VariableBuffer> createOrResizeBuffer(const std::string& name, size_t size, Device dev) {
    auto it = varBuffers_.find(name);
    if (it != varBuffers_.end()) {  // already set
      std::shared_ptr<VariableBuffer> buf = it->second;
      buf->resize(size);
      return buf;
    } else {
      return createBuffer(name, size, dev);
    }
  }

};

}
}