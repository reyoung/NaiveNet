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
    auto loss = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true, graph::kTENSOR_FLOAT32).get();
    addOp("cross_entropy", {input, label}, {loss});
    return loss;
  }

  graph::TensorAttr* mean(const std::string& paramPrefix,
                          graph::TensorAttr* input) {
    auto mean = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true, graph::kTENSOR_FLOAT32).get();
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
    auto paramTensor = graph_->createOrGetTensor(paramPrefix+".param.weight.0", {layerWidth, size}, true,
                                                 graph::kTENSOR_FLOAT32).get();
    SmallVec<graph::TensorAttr*> inputs = {input, paramTensor, nullptr};
    if (withBias) {
      auto biasTensor = graph_->createOrGetTensor(paramPrefix+".param.bias",{size, 1}, true,
                                                   graph::kTENSOR_FLOAT32).get();
      inputs.back() = biasTensor;
    }

    auto fcOpOut = graph_->createOrGetTensor(paramPrefix + "fc.output", {0}, true,
                                             graph::kTENSOR_FLOAT32).get();

    addOp("fc", inputs, {fcOpOut});

    auto finalOutput = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true,
                                                 graph::kTENSOR_FLOAT32).get();

    addOp(toString(act), {fcOpOut}, {finalOutput});
    return finalOutput;
  }

  // backward
  void backward(graph::TensorAttr* loss) {
    CHECK_EQ(loss->type_, graph::kTENSOR_FLOAT32) << "loss must be float32";
    CHECK_EQ(details::product(loss->dims_), 1)
        << "loss must be scalar, i.e. dim = 1";
    auto lossGradTensor = graph_->createOrGetTensor(loss->name_ + ".grad", {1, 1}, true, graph::kTENSOR_FLOAT32).get();
    lossGradTensor->specialResetFunction_ = [](graph::Tensor t) {
      *(float*)(t.buffer_->get()) = 1.0;
    };

    size_t fwdSize = graph_->ops_.size();

    for (size_t i = 0; i < fwdSize; ++i) {
      auto& op = graph_->ops_[fwdSize - i - 1];
      auto& opMeta = graph::OpMeta::gAllOpMeta_[op.type_];
      bool needGrad = false;
      auto needGradOrNull =
          [&needGrad](graph::TensorAttr* t) -> graph::TensorAttr* {
        if (t->needBackward_) {
          needGrad = true;
        }
            return t;
      };

      SmallVec<graph::TensorAttr*> outputs;
      outputs.resize(op.outputs_.size());
      std::transform(op.outputs_.begin(), op.outputs_.end(), outputs.begin(),
                     needGradOrNull);
      bool needGradOutput = needGrad;
      SmallVec<graph::TensorAttr*> inputs;
      inputs.resize(op.inputs_.size());
      std::transform(op.inputs_.begin(), op.inputs_.end(), inputs.begin(),
                     needGradOrNull);

      if (!needGrad) {
        continue;
      } else if (!needGradOutput && needGrad) {
        LOG(FATAL)
            << "All outputs of op don't contains grad, but need grad of input "
            << op.type_;
      }
      // else .. attach gradient op.
      // 1. get output g.
      SmallVec<graph::TensorAttr*> outputsGrad;
      for (auto& o : outputs) {
        if (!o->needBackward_) {
          outputsGrad.push_back(nullptr);
        } else {
          outputsGrad.push_back(graph_->createOrGetTensor(o->name_ + ".grad", o->dims_, true, o->type_).get());
        }
      }

      // 2. get input g.
      SmallVec<graph::TensorAttr*> inputsGrad;
      for (auto& ipt : inputs) {
        if (!ipt->needBackward_) {
          inputsGrad.push_back(nullptr);
        } else {
          inputsGrad.push_back(graph_->createOrGetTensor(ipt->name_ + ".grad", ipt->dims_, true, ipt->type_).get());
        }
      }

      // 3. attach ops.
      if (!opMeta.grad) {
        LOG(FATAL) << "Cannot perform backward, " << opMeta.type_
                   << " is not support backward";
      }

      auto gradOps = opMeta.grad(inputs, outputs, outputsGrad, inputsGrad);
      for (auto& o : gradOps) {
        graph_->ops_.push_back(o);
      }
    }
  };

 private:
  graph::Graph* graph_;
};
}
}

int main() {
  nnet::util::InitFunction::apply();
  nnet::graph::Graph g;
  constexpr size_t BATCH_SIZE = 1;
  auto buffer = nnet::memory::TensorBuffer::newBuffer<float>(
      "X", {BATCH_SIZE, 784}, nnet::memory::kDEVICE_CPU);
  nnet::api::GraphBuilder builder(&g);
  auto xTensor = g.createOrGetTensor("X", {BATCH_SIZE, 784}, false, nnet::graph::kTENSOR_FLOAT32).get();

  auto hidden = builder.fullyConnected("fc1", xTensor, 100, true);
  hidden = builder.fullyConnected("fc2", hidden, 100, true);
  auto prediction = builder.fullyConnected("prediction", hidden, 10, true,
                                           nnet::api::kSoftmax);
  auto labelBuffer = nnet::memory::TensorBuffer::newBuffer<int>(
      "Label", {BATCH_SIZE, 1}, nnet::memory::kDEVICE_CPU);
  auto labelTensor = g.createOrGetTensor("Label", {BATCH_SIZE, 1}, false, nnet::graph::kTENSOR_INT32).get();

  auto loss = builder.crossEntropy("xe_loss", prediction, labelTensor);
  auto avgLoss = builder.mean("avg_loss", loss);

  builder.backward(avgLoss);

  nnet::engine::NaiveEngine engine(g);
  engine.randomize();

  auto dataset = mnist::read_dataset_direct<std::vector, std::vector<uint8_t>>(
      "./3rdparty/mnist/");

  for (size_t i = 0; i < dataset.training_images.size() / BATCH_SIZE; ++i) {
    auto buf = (float*)buffer->get();
    auto labelBuf = (int*)labelBuffer->get();
    for (size_t j = 0; j < BATCH_SIZE; ++j) {
      auto& img = dataset.training_images[j + i * BATCH_SIZE];
      auto& lbl = dataset.training_labels[j + i * BATCH_SIZE];
      for (size_t k = 0; k < 784; ++k) {
        buf[j * 784 + k] = img[k];
      }
      labelBuf[j] = lbl;
    }
    nnet::engine::Tensor inputTensor;
    inputTensor.buffer_ = buffer;
    inputTensor.attr_ = xTensor;
    nnet::engine::castToEigenArray2DMutable(inputTensor) /= 255.0;
    engine.resetOrCreateGradient();
    engine.run(false);
//    engine.printMean();  // print mean grad of params
    nnet::engine::Tensor lossMemTensor;
    lossMemTensor.buffer_ =
        nnet::memory::TensorBuffer::gTensorBuffers.at(avgLoss->name_);
    lossMemTensor.attr_ = avgLoss;
    auto m = nnet::engine::castToEigenArray1D(lossMemTensor);
    LOG(INFO) << "Loss = " << *m.data();

  }

  return 0;
}