#include "Engine.h"
#include <random>
#include "misc/CastEigen.h"
namespace nnet {
namespace engine {
static thread_local std::default_random_engine* gGenerator;
std::default_random_engine& getGenerator() {
  if (gGenerator == nullptr) {
    gGenerator = new std::default_random_engine();
  }
  return *gGenerator;
}

#define castFN(__fn__) (std::bind(std::mem_fn(__fn__), this, std::placeholders::_1))

void NaiveEngine::randomize(Engine::NameMappingFN fn) const {
  if (!fn) {
    fn = castFN(&Engine::getParamInGraph);
  }

  this->accessVar(fn, [](Variable &var) {
    std::uniform_real_distribution<float> generator(-1.0, 1.0);
    LOG(INFO) << "Randomize " << var.attr_->name_;
    float *buf = (float *) var.buffer_->get();
    for (size_t i = 0; i < var.buffer_->getSize() / sizeof(float); ++i) {
      buf[i] = generator(getGenerator());
    }
  });
}

void NaiveEngine::resetOrCreateGradient(Engine::NameMappingFN fn) const {
  if (!fn) {
    fn = castFN(&Engine::getGradInGraph);
  }
  this->accessVar(fn, [](Variable &var) {
    if (var.attr_->specialResetFunction_) {
      var.attr_->specialResetFunction_(var);
    } else {
      auto varArr = eigen::cast<eigen::Vector>(var).array();
      varArr = 0.0;
    }
  });
}

static SmallVec<Variable> toVar(memory::Workspace &workspace, const SmallVec<graph::VariableAttrPtr> &vars) {
  SmallVec<Variable> retv;
  for (auto iptAttr : vars) {
    retv.emplace_back();
    auto& ipt = retv.back();
    ipt.attr_ = iptAttr;
    if (ipt.attr_ == nullptr) {
      ipt.buffer_ = nullptr;
    } else {
      ipt.buffer_ = workspace(ipt.attr_);
    }
  }
  return retv;
}

void NaiveEngine::run(bool debug) const {
  graph::Graph g = graph_;
  // Every mini-batch, shape could be changed.
  Map<std::string, Any> attrs;
  attrs["workspace"] = &this->workspace_;
  graph::compileGraph(&g, {"inferenceShape", "requestResource"}, attrs);

  for (auto& op : g.ops_) {
    auto& meta = graph::OpMeta::gAllOpMeta_[op.type_];
    auto ipt = toVar(workspace_, op.inputs_);
    auto opt = toVar(workspace_, op.outputs_);

    if (debug) {
      std::ostringstream sout;
      sout << " From: ";
      for (auto& i : op.inputs_) {
        if (i == nullptr) {
          sout << "nullptr";
        } else {
          sout << i->name_ << i->dims_ << " ";
        }
      }
      sout << "-> ";
      for (auto& o : op.outputs_) {
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
  this->accessVar(fn, [](Variable &var) {
    auto arr = eigen::cast<eigen::Vector>(var).array();
    float mean = arr.mean();
    LOG(INFO) << var.attr_->name_ << " mean = " << mean;
  });
}

void NaiveEngine::accessVar(NameMappingFN fn, std::function<void(Variable &)> tensorFN) const {
  for (auto& t : graph_.variables_) {
    auto var = fn(t.first);
    if (var != nullptr) {
      tensorFN(*var);
    }
  }
}
}
}