#pragma once
#include "TensorBuffer.h"
#include "graph/ComputationGraph.h"

namespace nnet {
namespace memory {

class Workspace final {
public:
  Map<std::string, std::shared_ptr<TensorBuffer>> tensorBuffers_;

  Workspace() = default;
  Workspace(const Workspace& o) = delete;

  std::shared_ptr<TensorBuffer> createBuffer(const std::string& name, size_t size, Device dev) {
    if (dev == kDEVICE_CPU) {
      auto buf = std::make_shared<CpuTensorBuffer>(size);
      tensorBuffers_.insert({name, buf});
      return buf;
    } else {
      LOG(FATAL) << "Not implemented";
      return nullptr;
    }
  }

  std::shared_ptr<TensorBuffer> operator()(const graph::TensorAttrPtr & attr) {
    size_t sz = details::product(attr->dims_);
    static_assert(sizeof(float) == sizeof(int), "");
    sz *= sizeof(float);
    return createOrResizeBuffer(attr->name_, sz, kDEVICE_CPU);
  }

  graph::Tensor getTensor(const graph::TensorAttrPtr & attr) {
    return {attr, this->operator()(attr)};
  }

  std::shared_ptr<TensorBuffer> createOrResizeBuffer(const std::string& name, size_t size, Device dev) {
    auto it = tensorBuffers_.find(name);
    if (it != tensorBuffers_.end()) {  // already set
      std::shared_ptr<TensorBuffer> buf = it->second;
      buf->resize(size);
      return buf;
    } else {
      return createBuffer(name, size, dev);
    }
  }

  template <typename T>
  std::shared_ptr<TensorBuffer> createOrResizeBufferT(const std::string& name, size_t size, Device dev) {
    return createOrResizeBuffer(name, size * sizeof(T), dev);
  }

  template <typename T>
  std::shared_ptr<TensorBuffer> createBufferT(const std::string& name, size_t size, Device dev) {
    return createBuffer(name, size*sizeof(T), dev);
  }
};

}
}