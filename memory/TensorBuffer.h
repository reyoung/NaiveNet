//
// Created by baidu on 2017/6/5.
//

#ifndef NAIVENET_TENSORBUFFER_H
#define NAIVENET_TENSORBUFFER_H
#include <easylogging++.h>
#include <cstddef>
#include <memory>
#include "misc/Error.h"
#include "misc/Typedef.h"

namespace nnet {
namespace memory {

enum Device : int { kDEVICE_CPU = 0, kDEVICE_GPU = 1, kNUM_DEVICES };

class TensorBuffer {
 public:
  TensorBuffer(size_t size, size_t capacity)
      : size_(size), capacity_(capacity) {}

  virtual ~TensorBuffer() {}
  virtual Device device() const = 0;
  virtual void copyFrom(const TensorBuffer& o, size_t elemSize = 0,
                        size_t sliceBegin = 0, size_t sliceEnd = 0) = 0;
  virtual void resize(size_t newSize) = 0;

  void* get() const { return buf_; }

  static Map<std::string, std::shared_ptr<TensorBuffer>> gTensorBuffers;

  template <typename T, typename Container = std::initializer_list<size_t>>
  static std::shared_ptr<TensorBuffer> newBuffer(const std::string& name,
                                                 Container dims, Device dev);

  template <typename T, typename Container = std::initializer_list<size_t>>
  static std::shared_ptr<TensorBuffer> createOrResizeBuffer(
      const std::string& name, Container dims, Device dev = kDEVICE_CPU) {
    auto it = memory::TensorBuffer::gTensorBuffers.find(name);
    if (it == memory::TensorBuffer::gTensorBuffers.end()) {
      return memory::TensorBuffer::newBuffer<T>(name, dims, dev);
    } else {
      it->second->resize(sizeof(T) * details::product(dims));
      return it->second;
    }
  }

  size_t getSize() const { return size_; }

 protected:
  void* buf_{nullptr};
  size_t size_{0};
  size_t capacity_{0};
};

using TensorBufferPtr = std::shared_ptr<TensorBuffer>;

class CpuTensorBuffer : public TensorBuffer {
 public:
  CpuTensorBuffer() = default;

  CpuTensorBuffer(size_t size) : TensorBuffer(size, size) {
    CHECK_EQ(posix_memalign(&buf_, 32UL, size), 0);
  }

  ~CpuTensorBuffer() { free(buf_); }

  Device device() const override { return kDEVICE_CPU; }

  void resize(size_t newSize) override {
    if (newSize > capacity_) {
      void* tmp;
      CHECK_EQ(posix_memalign(&tmp, 32UL, newSize), 0);
      free(buf_);
      buf_ = tmp;
    }
    size_ = newSize;
  }

  void copyFrom(const TensorBuffer& o, size_t elemSize = 0,
                size_t sliceBegin = 0, size_t sliceEnd = 0) override {
    LOG(FATAL) << "Not Implemented";
  }
};

template <typename T, typename Container>
std::shared_ptr<TensorBuffer> TensorBuffer::newBuffer(const std::string& name,
                                                      Container dims,
                                                      Device dev) {
  CHECK_EQ(gTensorBuffers.find(name), gTensorBuffers.end());
  size_t prod = sizeof(T);
  prod *= details::product(dims);

  if (dev == kDEVICE_CPU) {
    gTensorBuffers[name].reset(new CpuTensorBuffer(prod));
    return gTensorBuffers[name];
  } else {
    LOG(FATAL) << "Not Implemented";
  }
  return nullptr;
}
}
}

#endif  // NAIVENET_TENSORBUFFER_H
