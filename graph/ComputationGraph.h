#pragma once
#include <stdint.h>
#include <functional>
#include <typeinfo>
#include "ComputationGraph.h"
#include "memory/VariableBuffer.h"
#include "misc/Error.h"
#include "misc/Typedef.h"

namespace nnet {
namespace graph {
using memory::Device;
using memory::kDEVICE_GPU;
using memory::kDEVICE_CPU;
using memory::kNUM_DEVICES;

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
    return add([=](T* attr, bool set) {
      if (!set) {
        *attr = defaultVal;
      }
    });
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
    for (auto& f : constraints_) {
      f(attr, alreadySet);
    }
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
  static std::shared_ptr<AttributeMeta> create(const std::string& name, const std::string& description) {
    return std::make_shared<AttributeMeta>(name, description, typeid(T),
                                           std::unique_ptr<BaseConstraints>(new Constraints<T>()));
  }

  inline AttributeMeta(const std::string& name, const std::string& desc, const std::type_info& type,
                       std::unique_ptr<BaseConstraints>&& cons)
      : name_(name), description_(desc), type_(type), constraints_(std::move(cons)) {}
};

enum VariableType : size_t { kFLOAT32 = 0, kINT32 = 1 };
class VariableAttr;

class Variable final {
 public:
  std::shared_ptr<VariableAttr> attr_;
  std::shared_ptr<memory::VariableBuffer> buffer_;
};

class VariableAttr final {
 public:
  using InitializeFN = std::function<void(Variable)>;
  VariableAttr() = default;
  VariableAttr(const std::string& name, const SmallVec<size_t>& dims, VariableType type, bool needBackward)
      : name_(name), needBackward_(needBackward), dims_(dims), type_(type) {}

  bool sameNameAndType(const VariableAttr& attr) const { return attr.name_ == name_ && attr.type_ == type_; }

  bool operator==(const VariableAttr& attr) const { return sameNameAndType(attr) && attr.dims_ == dims_; }

  bool operator!=(const VariableAttr& attr) const { return !this->operator==(attr); }

  std::string name_;
  bool needBackward_{true};
  SmallVec<size_t> dims_;
  VariableType type_;
  InitializeFN specialResetFunction_;  // when apply reset to vars, default is
                                       // reset to zero.
};

using VariableAttrPtr = std::shared_ptr<VariableAttr>;

class Op final {
 public:
  std::string type_;
  SmallVec<VariableAttrPtr> inputs_;
  SmallVec<VariableAttrPtr> outputs_;
  Map<std::string, Any> attrs_;

  Op() = default;
  Op(const std::string& type, const SmallVec<VariableAttrPtr>& inputs, const SmallVec<VariableAttrPtr>& outputs,
     const Map<std::string, Any>& attr = Map<std::string, Any>())
      : type_(type), inputs_(inputs), outputs_(outputs), attrs_(attr) {}
};

class OpMeta final {
 public:
  using ShapeInfererFN =
      std::function<void(const SmallVec<VariableAttrPtr>& inputs, const SmallVec<VariableAttrPtr>& outputs)>;
  using RunOnDeviceFN = std::function<void(const SmallVec<Variable>& inputs, SmallVec<Variable>& outputs,
                                           const Map<std::string, Any> attrs)>;

  using GradFN = std::function<SmallVec<Op>(
      const SmallVec<VariableAttrPtr>& inputs, const SmallVec<VariableAttrPtr>& outputs,
      const SmallVec<VariableAttrPtr>& outputsGrad, const SmallVec<VariableAttrPtr>& inputsGrad)>;

  using GradVariablesOp = std::function<void(const SmallVec<VariableAttrPtr>& I, const SmallVec<VariableAttrPtr>& O,
                                             SmallVec<VariableAttrPtr>* IG, SmallVec<VariableAttrPtr>* OG)>;

  std::string type_;
  ShapeInfererFN shapeInferer_;
  SmallVec<std::shared_ptr<AttributeMeta>> attrMeta_;
  RunOnDeviceFN kernels[kNUM_DEVICES];
  GradFN grad_;
  GradVariablesOp gradVars_{[](const SmallVec<VariableAttrPtr>& I, const SmallVec<VariableAttrPtr>& O,
                               SmallVec<VariableAttrPtr>* OG, SmallVec<VariableAttrPtr>* IG) {
    auto transformImpl = [](const VariableAttrPtr& ptr) -> VariableAttrPtr {
      if (!ptr) {
        return nullptr;
      } else if (ptr->needBackward_) {
        auto retv = std::make_shared<VariableAttr>();
        *retv = *ptr;
        retv->name_ += ".grad";
        return retv;
      } else {
        return nullptr;
      }
    };

    IG->resize(I.size());
    OG->resize(O.size());
    std::transform(I.begin(), I.end(), IG->begin(), transformImpl);
    std::transform(O.begin(), O.end(), OG->begin(), transformImpl);
  }};

  static Map<std::string, OpMeta> gAllOpMeta_;
};

class Graph final {
 public:
  Map<std::string, VariableAttrPtr> variables_;
  SmallVecN<Op, 10> ops_;

  template <bool failWhenMismatchDims = false>
  VariableAttrPtr createOrResizeVar(const std::string& name, const SmallVec<size_t>& dim, bool need_backward,
                                    VariableType type) {
    auto attr = std::make_shared<VariableAttr>(name, dim, type, need_backward);
    return createOrResizeVar<failWhenMismatchDims>(attr);
  }

  template <bool failWhenMismatchDims = false>
  VariableAttrPtr createOrResizeVar(const VariableAttrPtr& ptr) {
    auto it = variables_.find(ptr->name_);
    if (it == variables_.end()) {
      variables_.insert({ptr->name_, ptr});
      return ptr;
    } else {
      if (failWhenMismatchDims) {
        CHECK_EQ(*it->second, *ptr) << "Dimension mismatch";
      } else {
        CHECK(ptr->sameNameAndType(*it->second));
        it->second->dims_ = ptr->dims_;
      }
      return it->second;
    }
  }
};

using CompileGraphFN = std::function<void(Graph&, const Map<std::string, Any>&)>;
extern Map<std::string, CompileGraphFN>& compilers();
inline void compileGraph(Graph* g, const SmallVec<std::string>& stages,
                         const Map<std::string, Any>& attrs = Map<std::string, Any>()) {
  for (auto& key : stages) {
    compilers().at(key)(*g, attrs);
  }
}
}
}
