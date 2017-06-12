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
  TensorBuffer(size_t size, size_t capacity) : size_(size), capacity_(capacity) {}

  virtual ~TensorBuffer() {}
  virtual Device device() const = 0;
  virtual void resize(size_t newSize) = 0;

  void* get() const { return buf_; }

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

  CpuTensorBuffer(size_t size) : TensorBuffer(size, size) { CHECK_EQ(posix_memalign(&buf_, 32UL, size), 0); }

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
};

}
}

#endif  // NAIVENET_TENSORBUFFER_H
