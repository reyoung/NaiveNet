#pragma once
#include <stdint.h>
#include <functional>
#include <typeinfo>
#include "ComputationGraph.h"
#include "Error.h"
#include "TensorBuffer.h"
#include "Typedef.h"

namespace nnet {
namespace graph {
using memory::Device;
using memory::kDEVICE_GPU;
using memory::kDEVICE_CPU;
using memory::kDEVICE_DEBUG;
using memory::kNUM_DEVICES;

namespace error {
class DefaultValueHasBeenSet : public Error {
 public:
  DefaultValueHasBeenSet() : Error("Default value has been set") {}
};
}

class BaseConstraints {
 public:
  virtual void check(const std::string& name, Map<std::string, Any>* attrs) = 0;
};

template <typename T>
class Constraints final : public BaseConstraints {
 public:
  using ConstraintFN = std::function<void(T* attr, bool alreadySet)>;

  Constraints<T>& add(ConstraintFN fn) {
    constraints_.push_back(fn);
    return *this;
  }

  Constraints<T>& defaultValue(const T& defaultVal) {
    return add([=](T* attr, bool alreadySet) {
      if (alreadySet) throw error::DefaultValueHasBeenSet();
      *attr = defaultVal;
    });
  }

  void check(T* attr, bool alreadySet) const throw(Error) {
    for (auto& f : constraints_) {
      f(attr, alreadySet);
    }
  }

  void check(const std::string& name, Map<std::string, Any>* attrs) override {
    bool alreadySet = attrs->find(name) != attrs->end();
    Any* ptr;
    if (!alreadySet) {
      (*attrs)[name] = T();
    }
    ptr = &attrs->find(name)->second;

    T* attr = any_cast<T>(ptr);
    CHECK_NE(attr, nullptr);
    this->check(attr, alreadySet);
  }

 private:
  SmallVec<ConstraintFN> constraints_;
};

class AttributeMeta final {
 public:
  std::string name_;
  std::string description_;
  const std::type_info& type_;
  std::unique_ptr<BaseConstraints> constraints_;

  template <typename T>
  Constraints<T>& constraints() {
    auto ptr = reinterpret_cast<Constraints<T>*>(constraints_.get());
    return *ptr;
  }

  template <typename T>
  static std::shared_ptr<AttributeMeta> create(const std::string& name,
                                               const std::string& description) {
    return std::make_shared<AttributeMeta>(
        name, description, typeid(T),
        std::unique_ptr<BaseConstraints>(new Constraints<T>()));
  }

  inline AttributeMeta(const std::string& name, const std::string& desc,
                       const std::type_info& type,
                       std::unique_ptr<BaseConstraints>&& cons)
      : name_(name),
        description_(desc),
        type_(type),
        constraints_(std::move(cons)) {}
};

enum TensorType : size_t { kTENSOR_FLOAT32 = 0, kTENSOR_INT32 = 1 };
class TensorAttr;

class Tensor final {
 public:
  TensorAttr* attr_;
  std::shared_ptr<memory::TensorBuffer> buffer_;
};

class TensorAttr final {
 public:
  using InitializeFN = std::function<void(Tensor)>;
  TensorAttr() = default;
  TensorAttr(const std::string& name,
             const SmallVec<size_t>& dims,
             TensorType type,
             bool needBackward): name_(name), needBackward_(needBackward), dims_(dims), type_(type) {}

  bool sameNameAndType(const TensorAttr& attr) const {
    return attr.name_ == name_ && attr.type_ == type_;
  }

  bool operator == (const TensorAttr& attr) const {
    return sameNameAndType(attr) && attr.dims_ == dims_;
  }

  bool operator != (const TensorAttr& attr) const {
    return !this->operator==(attr);
  }

  std::string name_;
  bool needBackward_{true};
  SmallVec<size_t> dims_;
  TensorType type_;
  InitializeFN specialResetFunction_;  // when apply reset to tensor, default is reset to zero.
};



class Op final {
 public:
  std::string type_;
  SmallVec<TensorAttr*> inputs_;
  SmallVec<TensorAttr*> outputs_;
  Map<std::string, Any> attrs_;
};

class OpMeta final {
 public:
  using ShapeInfererFN =
      std::function<void(const SmallVec<TensorAttr*>& inputs,
                         const SmallVec<TensorAttr*>& outputs)>;
  using RunOnDeviceFN = std::function<void(const SmallVec<Tensor>& inputs,
                                           SmallVec<Tensor>& outputs,
                                           const Map<std::string, Any> attrs)>;

  using GradFN = std::function<SmallVec<Op>(
      const SmallVec<TensorAttr*>& inputs, const SmallVec<TensorAttr*>& outputs,
      const SmallVec<TensorAttr*>& outputsGrad,
      const SmallVec<TensorAttr*>& inputsGrad)>;

  std::string type_;
  ShapeInfererFN shapeInferer_;
  SmallVec<std::shared_ptr<AttributeMeta>> attrMeta_;
  RunOnDeviceFN kernels[kNUM_DEVICES];
  GradFN grad;

  static Map<std::string, OpMeta> gAllOpMeta_;
};


using TensorAttrPtr = std::shared_ptr<TensorAttr>;

class Graph final {
 public:
  Map<std::string, TensorAttrPtr> tensors_;
  SmallVecN<Op, 10> ops_;

  template <bool failWhenMismatchDims=false>
  TensorAttrPtr createOrGetTensor(const std::string& name,
                           const SmallVec<size_t> & dim,
                           bool need_backward,
                           TensorType type) {
    auto attr = std::make_shared<TensorAttr>(name, dim, type, need_backward);
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
      tensors_.insert({attr->name_, attr});
    } else {
      if (failWhenMismatchDims) {
        CHECK_EQ(*it->second, *attr) << "Dimension mismatch";
      } else {
        CHECK(attr->sameNameAndType(*it->second));
        it->second->dims_ = attr->dims_;
      }
      attr = it->second;
    }
    return attr;
  }
};

using CompileGraphFN = std::function<Graph (const Graph&)>;
extern Map<std::string, CompileGraphFN >& compilers();

}
}
