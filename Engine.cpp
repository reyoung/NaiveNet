#include "Engine.h"
#include <random>
namespace nnet {
namespace engine {
static thread_local std::default_random_engine* gGenerator;
std::default_random_engine& getGenerator() {
  if (gGenerator == nullptr) {
    gGenerator = new std::default_random_engine();
  }
  return *gGenerator;
}

void NaiveEngine::randomize(Engine::NameMappingFN fn) const {
  if (!fn) {
    fn = std::bind(std::mem_fn(&Engine::getParamInGraph), this,
                   std::placeholders::_1);
  }

  std::uniform_real_distribution<float> generator(-1.0, 1.0);

  for (auto& t : graph_.tensors_) {
    auto tensor = fn(t.first);
    if (tensor != nullptr) {
      LOG(INFO) << "Randomize " << t.first;
      float* buf = (float*)tensor->buffer_->get();
      for (size_t i = 0; i < tensor->buffer_->getSize() / sizeof(float); ++i) {
        buf[i] = generator(getGenerator());
      }
    }
  }
}

static SmallVec<Tensor> toTensor(const SmallVec<graph::TensorAttr*>& tensors) {
  SmallVec<Tensor> retv;
  for (auto iptAttr : tensors) {
    retv.emplace_back();
    auto& ipt = retv.back();
    ipt.attr_ = iptAttr;
    if (ipt.attr_->type_ == graph::kTENSOR_FLOAT32) {
      ipt.buffer_ = memory::TensorBuffer::tryAllocBuffer<float>(iptAttr->name_,
                                                                iptAttr->dims_);
    } else if (ipt.attr_->type_ == graph::kTENSOR_INT32) {
      ipt.buffer_ = memory::TensorBuffer::tryAllocBuffer<int>(iptAttr->name_,
                                                              iptAttr->dims_);
    }
  }
  return retv;
}

void NaiveEngine::run(bool debug) const {
  for (auto& op : graph_.ops_) {
    auto& meta = graph::OpMeta::gAllOpMeta_[op.type_];
    auto ipt = toTensor(op.inputs_);
    auto opt = toTensor(op.outputs_);

    std::ostringstream sout;
    sout << " From: ";
    for (auto& i : op.inputs_) {
      sout << i->name_ << i->dims_ << " ";
    }
    sout << "-> ";
    for (auto& o : op.outputs_) {
      sout << o->name_ << o->dims_ << " ";
    }

    LOG(INFO) << "Performing " << op.type_ << sout.str();
    if (!debug) {
      meta.kernels[graph::kDEVICE_CPU](ipt, opt, op.attrs_);
    }
  }
}
}
}