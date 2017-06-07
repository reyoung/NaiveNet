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

class TensorAttr final {
 public:
  std::string name_;
  bool need_backward_{false};
  SmallVec<size_t> dims_;
  TensorType type_;
};

class Tensor final {
 public:
  TensorAttr* attr_;
  std::shared_ptr<memory::TensorBuffer> buffer_;
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

class GraphMeta final {
 public:
};

class Graph final {
 public:
  Map<std::string, TensorAttr> tensors_;
  SmallVecN<Op, 10> ops_;
};
}
}
