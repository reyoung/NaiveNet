#include <iostream>
// should use glog instead of elpp, because we could throw a Error when
// log(Fatal)
#include <easylogging++.h>
#include <mnist/mnist_reader.hpp>
#include "ComputationGraph.h"
#include "Engine.h"
#include "Error.h"
#include "Register.h"

INITIALIZE_EASYLOGGINGPP

namespace nnet {
namespace api {
enum ActivationType { kSigmoid, kSoftmax };

const char* toString(ActivationType act) {
  switch (act) {
    case kSigmoid:
      return "sigmoid";
    case kSoftmax:
      return "softmax";
    default:
      LOG(FATAL) << "Not supported act " << act;
  }
}

class GraphBuilder {
 public:
  explicit inline GraphBuilder(graph::Graph* g) : graph_(g) {}

  template <typename T, typename Container = std::initializer_list<size_t >>
  graph::TensorAttr* addTensor(const std::string& name,
                               Container dims,
                               bool need_grad) {
    if (typeid(T) == typeid(float)) {
      return addTensorImpl(name, dims, need_grad, graph::kTENSOR_FLOAT32);
    } else if (typeid(T) == typeid(int)) {
      return addTensorImpl(name, dims, need_grad, graph::kTENSOR_INT32);
    } else {
      throw Error("Not supported type of tensor %s", typeid(T).name());
    }
  }

  void addOp(const std::string& type,
             const SmallVec<graph::TensorAttr*>& inputs,
             const SmallVec<graph::TensorAttr*>& outputs,
             const Map<std::string, Any>& attrs = Map<std::string, Any>()) {
    this->graph_->ops_.emplace_back();
    graph::Op& op = this->graph_->ops_.back();
    op.type_ = type;
    op.attrs_ = attrs;
    op.inputs_ = inputs;
    op.outputs_ = outputs;
    graph::OpMeta& meta = graph::OpMeta::gAllOpMeta_[op.type_];
    meta.shapeInferer_(inputs, outputs);
    for (auto& attrMeta : meta.attrMeta_) {
      attrMeta->constraints_->check(attrMeta->name_, &op.attrs_);
    }
  }

  graph::TensorAttr* crossEntropy(const std::string& paramPrefix,
                                  graph::TensorAttr* input,
                                  graph::TensorAttr* label) {
    auto loss = addTensor<float>(paramPrefix + ".output", {0}, true);
    addOp("cross_entropy", {input, label}, {loss});
    return loss;
  }

  graph::TensorAttr* mean(const std::string& paramPrefix, graph::TensorAttr* input) {
    auto mean = addTensor<float >(paramPrefix + ".output", {0}, true);
    addOp("mean", {input}, {mean});
    return mean;
  }


  graph::TensorAttr* fullyConnected(const std::string& paramPrefix,
                                    graph::TensorAttr* input, size_t size,
                                    bool withBias = true,
                                    const ActivationType& act = kSigmoid,
                                    bool allocParam = true) {
    CHECK_EQ(input->dims_.size(), 2UL);
    auto layerWidth = input->dims_[1];

    if (allocParam) {
      memory::TensorBuffer::tryAllocBuffer<float>(
          paramPrefix + ".param.weight.0", {layerWidth, size});
      if (withBias) {
        memory::TensorBuffer::tryAllocBuffer<float>(paramPrefix + ".param.bias",
                                                    {size});
      }
    }

    auto paramTensor = addTensor<float>(paramPrefix + ".param.weight.0",
                                        {layerWidth, size}, true);
    auto tmpMulOut = addTensor<float>(paramPrefix + ".tmp.0", {0}, true);
    addOp("mul", {input, paramTensor}, {tmpMulOut});

    if (withBias) {
      auto biasTensor =
          addTensor<float>(paramPrefix + ".param.bias", {size, 1}, true);
      auto out = addTensor<float>(paramPrefix + ".tmp.1", {0}, true);
      addOp("add_bias", {tmpMulOut, biasTensor}, {out});
      tmpMulOut = out;
    }

    auto finalOutput = addTensor<float>(paramPrefix + ".output", {0}, true);
    addOp(toString(act), {tmpMulOut}, {finalOutput});
    return finalOutput;
  }

  // backward
  void backward(graph::TensorAttr* loss) {
    CHECK_EQ(loss->type_, graph::kTENSOR_FLOAT32) << "loss must be float32";
    CHECK_EQ(details::product(loss->dims_), 1) << "loss must be scalar, i.e. dim = 1";

    size_t fwdSize = graph_->ops_.size();


    for (size_t i=0; i<fwdSize; ++i) {
      auto& op = graph_->ops_[fwdSize - i - 1];
      auto& opMeta = graph::OpMeta::gAllOpMeta_[op.type_];
      bool needGrad = false;
      auto needGradOrNull = [&needGrad](graph::TensorAttr* t) -> graph::TensorAttr* {
        if (t->need_backward_) {
          needGrad = true;
          return t;
        } else {
          return nullptr;
        }
      };

      SmallVec <graph::TensorAttr*> outputs;
      outputs.resize(op.outputs_.size());
      std::transform(op.outputs_.begin(), op.outputs_.end(), outputs.begin(), needGradOrNull);
      bool needGradOutput = needGrad;
      SmallVec <graph::TensorAttr*> inputs;
      inputs.resize(op.inputs_.size());
      std::transform(op.inputs_.begin(), op.inputs_.end(), inputs.begin(), needGradOrNull);

      if (!needGrad) {
        continue;
      } else if (!needGradOutput && needGrad) {
        LOG(FATAL) << "All outputs of op don't contains grad, but need grad of input " << op.type_;
      }
      // else .. attach gradient op.
      // 1. get output g.
      SmallVec <graph::TensorAttr*> outputsGrad;
      for (auto& o : outputs) {
        if (o == nullptr) {
          outputsGrad.push_back(nullptr);
        } else {
          outputsGrad.push_back(this->addTensorImpl(o->name_+".grad", o->dims_, true, o->type_));
        }
      }

      // 2. get input g.
      SmallVec <graph::TensorAttr*> inputsGrad;
      for (auto& ipt : inputs) {
        if (ipt == nullptr) {
          inputsGrad.push_back(nullptr);
        } else {
          inputsGrad.push_back(this->addTensorImpl(ipt->name_+".grad", ipt->dims_, true, ipt->type_));
        }
      }

      // 3. attach ops.
      if (!opMeta.grad) {
        LOG(FATAL) << "Cannot perform backward, " << opMeta.type_ << " is not support backward";
      }

      auto gradOps = opMeta.grad(inputs, outputs, outputsGrad, inputsGrad);
      for (auto & o : gradOps) {
        graph_->ops_.push_back(o);
      }
    }
  };


 private:
  template <typename Container>
  graph::TensorAttr* addTensorImpl(const std::string& name,
                                   Container dim,
                                   bool need_grad,
                                   graph::TensorType type) {
    auto it = graph_->tensors_.find(name);
    if (it == graph_->tensors_.end()) {
      graph::TensorAttr tensor;
      tensor.name_ = name;
      tensor.dims_ = dim;
      tensor.need_backward_ = need_grad;
      tensor.type_ = type;
      graph_->tensors_[name] = tensor;
      return &graph_->tensors_[name];
    } else {
      auto dimIt1 = it->second.dims_.begin();
      auto dimIt2 = dim.begin();
      for (; dimIt1 != it->second.dims_.end(); ++dimIt1, ++dimIt2) {
        CHECK_EQ(*dimIt1, *dimIt2);
      }
      CHECK_EQ(it->second.need_backward_, need_grad);
      CHECK_EQ(it->second.type_, type);
      return &it->second;
    }
  }

 private:
  graph::Graph* graph_;
};
}
}

int main() {
  nnet::util::InitFunction::apply();
  nnet::graph::Graph g;
  constexpr size_t BATCH_SIZE=1000;
  auto buffer = nnet::memory::TensorBuffer::newBuffer<float>(
      "X", {BATCH_SIZE, 784}, nnet::memory::kDEVICE_CPU);
  nnet::api::GraphBuilder builder(&g);
  auto xTensor = builder.addTensor<float>("X", {BATCH_SIZE, 784}, false);

  auto hidden = builder.fullyConnected("fc1", xTensor, 100, true);
  hidden = builder.fullyConnected("fc2", hidden, 100, true);
  auto prediction = builder.fullyConnected("prediction", hidden, 10, true,
                                           nnet::api::kSoftmax);

  auto labelBuffer = nnet::memory::TensorBuffer::newBuffer<int>(
      "Label", {BATCH_SIZE, 1}, nnet::memory::kDEVICE_CPU);
  auto labelTensor = builder.addTensor<int>("Label", {BATCH_SIZE, 1}, false);
  auto loss = builder.crossEntropy("xe_loss", prediction, labelTensor);
  auto avgLoss = builder.mean("avg_loss", loss);

//  builder.backward(avgLoss);

  nnet::engine::NaiveEngine engine(g);
  engine.randomize();

  auto dataset = mnist::read_dataset_direct<std::vector, std::vector<uint8_t >>(
      "./3rdparty/mnist/"
  );

  for (size_t i=0; i<dataset.training_images.size() / BATCH_SIZE; ++i) {
    auto buf = (float*) buffer->get();
    auto labelBuf = (int*) labelBuffer->get();
    for (size_t j=0; j < BATCH_SIZE; ++j) {
      auto& img = dataset.training_images[j + i*BATCH_SIZE];
      auto& lbl = dataset.training_labels[j + i*BATCH_SIZE];
      for (size_t k = 0; k<784; ++k) {
        buf[j*784+k] = img[k];
      }
      labelBuf[j] = lbl;
    }
    nnet::engine::Tensor inputTensor;
    inputTensor.buffer_ = buffer;
    inputTensor.attr_ = xTensor;
    nnet::engine::castToEigenArray2DMutable(inputTensor) /= 255.0;
    engine.run(false);

    nnet::engine::Tensor lossMemTensor;
    lossMemTensor.buffer_ = nnet::memory::TensorBuffer::gTensorBuffers.at(avgLoss->name_);
    lossMemTensor.attr_ = avgLoss;
    auto m = nnet::engine::castToEigenArray1D(lossMemTensor);
    LOG(INFO) << "Loss = " << *m.data();
  }




  return 0;
}