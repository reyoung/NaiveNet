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

#define castFN(__fn__) (std::bind(std::mem_fn(__fn__), this, std::placeholders::_1));

void NaiveEngine::randomize(Engine::NameMappingFN fn) const {
  if (!fn) {
    fn = castFN(&Engine::getParamInGraph);
  }

  this->accessTensor(fn, [](Tensor& tensor){
    std::uniform_real_distribution<float> generator(-1.0, 1.0);
    LOG(DEBUG) << "Randomize " << tensor.attr_->name_;
    float* buf = (float*)tensor.buffer_->get();
    for (size_t i = 0; i < tensor.buffer_->getSize() / sizeof(float); ++i) {
      buf[i] = generator(getGenerator());
    }
  });
}

void NaiveEngine::resetOrCreateGradient(Engine::NameMappingFN fn) const {
  if (!fn) {
    fn = castFN(&Engine::getGradInGraph);
  }
  this->accessTensor(fn, [](Tensor& tensor) {
//      LOG(DEBUG) << "Resetting buffer " << tensor.attr_->name_;
      if (tensor.attr_->specialResetFunction_) {
//        LOG(DEBUG) << "Special init gradient buffer " << tensor.attr_->name_;
        tensor.attr_->specialResetFunction_(tensor);
      } else {
        auto tensorArray = engine::castToEigenArray1DMutable(tensor);
        tensorArray = 0.0;
      }
  });
}

static SmallVec<Tensor> toTensor(const SmallVec<graph::TensorAttrPtr>& tensors) {
  SmallVec<Tensor> retv;
  for (auto iptAttr : tensors) {
    retv.emplace_back();
    auto& ipt = retv.back();
    ipt.attr_ = iptAttr;
    if (ipt.attr_ == nullptr) {
      ipt.buffer_ = nullptr;
    } else if (ipt.attr_->type_ == graph::kTENSOR_FLOAT32) {
      ipt.buffer_ = memory::TensorBuffer::createOrResizeBuffer<float>(iptAttr->name_,
                                                                      iptAttr->dims_);
    } else if (ipt.attr_->type_ == graph::kTENSOR_INT32) {
      ipt.buffer_ = memory::TensorBuffer::createOrResizeBuffer<int>(iptAttr->name_,
                                                                    iptAttr->dims_);
    }
  }
  return retv;
}

void NaiveEngine::run(bool debug) const {
  graph::Graph g = graph_;
  // Every mini-batch, shape could be changed.
  graph::compileGraph(&g, {"inferenceShape", "requestResource"});

  for (auto& op : g.ops_) {
    auto& meta = graph::OpMeta::gAllOpMeta_[op.type_];
    auto ipt = toTensor(op.inputs_);
    auto opt = toTensor(op.outputs_);

    if (debug) {
      std::ostringstream sout;
      sout << " From: ";
      for (auto &i : op.inputs_) {
        if (i == nullptr) {
          sout << "nullptr";
        } else {
          sout << i->name_ << i->dims_ << " ";
        }
      }
      sout << "-> ";
      for (auto &o : op.outputs_) {
        if (o == nullptr) {
          sout << "nullptr";
        } else {
          sout << o->name_ << o->dims_ << " ";
        }
      }
      LOG(DEBUG) << "Performing " << op.type_ << sout.str();
    }

    meta.kernels[graph::kDEVICE_CPU](ipt, opt, op.attrs_);
  }
}

void NaiveEngine::printMean(NameMappingFN fn) const {
  if (!fn) {
    fn = castFN(&Engine::getGradInGraph);
  }
  this->accessTensor(fn, [](Tensor& tensor) {
    auto arr = castToEigenArray1D(tensor);
    float mean = arr.mean();
    LOG(INFO) << tensor.attr_->name_ << " mean = " << mean;
  });
}

void NaiveEngine::accessTensor(NameMappingFN fn, std::function<void(Tensor &)> tensorFN) const {
  for (auto& t : graph_.tensors_) {
    auto tensor = fn(t.first);
    if (tensor != nullptr) {
      tensorFN(*tensor);
    }
  }
}


}

}