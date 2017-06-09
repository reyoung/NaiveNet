#include <iostream>
#include <Eigen/SparseCore>
// should use glog instead of elpp, because we could throw a Error when
// log(Fatal)
#include <easylogging++.h>
#include <mnist/mnist_reader.hpp>
#include "engine/Engine.h"
#include "graph/ComputationGraph.h"
#include "misc/Error.h"
#include "misc/CastEigen.h"
#include "misc/InitFunction.h"

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

  void addOp(const std::string& type, const SmallVec<graph::TensorAttrPtr>& inputs,
             const SmallVec<graph::TensorAttrPtr>& outputs,
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

  graph::TensorAttrPtr crossEntropy(const std::string& paramPrefix, graph::TensorAttrPtr input,
                                    graph::TensorAttrPtr label) {
    auto loss = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true, graph::kTENSOR_FLOAT32);
    addOp("cross_entropy", {input, label}, {loss});
    return loss;
  }

  graph::TensorAttrPtr mean(const std::string& paramPrefix, graph::TensorAttrPtr input) {
    auto mean = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true, graph::kTENSOR_FLOAT32);
    addOp("mean", {input}, {mean});
    return mean;
  }

  graph::TensorAttrPtr fullyConnected(const std::string& paramPrefix, graph::TensorAttrPtr input, size_t size,
                                      bool withBias = true, const ActivationType& act = kSigmoid,
                                      bool allocParam = true) {
    CHECK_EQ(input->dims_.size(), 2UL);
    auto layerWidth = input->dims_[1];

    if (allocParam) {
      memory::TensorBuffer::createOrResizeBuffer<float>(paramPrefix + ".param.weight.0", {layerWidth, size});
      if (withBias) {
        memory::TensorBuffer::createOrResizeBuffer<float>(paramPrefix + ".param.bias", {size});
      }
    }
    auto paramTensor =
        graph_->createOrGetTensor(paramPrefix + ".param.weight.0", {layerWidth, size}, true, graph::kTENSOR_FLOAT32);
    SmallVec<graph::TensorAttrPtr> inputs = {input, paramTensor, nullptr};
    if (withBias) {
      auto biasTensor = graph_->createOrGetTensor(paramPrefix + ".param.bias", {size, 1}, true, graph::kTENSOR_FLOAT32);
      inputs.back() = biasTensor;
    }

    auto fcOpOut = graph_->createOrGetTensor(paramPrefix + "fc.output", {0}, true, graph::kTENSOR_FLOAT32);

    addOp("fc", inputs, {fcOpOut});

    auto finalOutput = graph_->createOrGetTensor(paramPrefix + ".output", {0}, true, graph::kTENSOR_FLOAT32);

    addOp(toString(act), {fcOpOut}, {finalOutput});
    return finalOutput;
  }

  // backward
  void backward(graph::TensorAttrPtr loss) {
    Map<std::string, Any> attrs;
    attrs.insert({"loss_name", loss->name_});
    graph::compileGraph(graph_, {"backward"}, attrs);
  };

 private:
  graph::Graph* graph_;
};
}
}

static void TrainMnistOnePass(bool printGradMean = false) {
  nnet::graph::Graph g;
  constexpr size_t BATCH_SIZE = 1000;
  auto buffer = nnet::memory::TensorBuffer::newBuffer<float>("X", {BATCH_SIZE, 784}, nnet::memory::kDEVICE_CPU);
  nnet::api::GraphBuilder builder(&g);
  auto xTensor = g.createOrGetTensor("X", {BATCH_SIZE, 784}, false, nnet::graph::kTENSOR_FLOAT32);

  auto hidden = builder.fullyConnected("fc1", xTensor, 100, true);
  hidden = builder.fullyConnected("fc2", hidden, 100, true);
  auto prediction = builder.fullyConnected("prediction", hidden, 10, true, nnet::api::kSoftmax);
  auto labelBuffer = nnet::memory::TensorBuffer::newBuffer<int>("Label", {BATCH_SIZE, 1}, nnet::memory::kDEVICE_CPU);
  auto labelTensor = g.createOrGetTensor("Label", {BATCH_SIZE, 1}, false, nnet::graph::kTENSOR_INT32);

  auto loss = builder.crossEntropy("xe_loss", prediction, labelTensor);
  auto avgLoss = builder.mean("avg_loss", loss);

  builder.backward(avgLoss);
  nnet::graph::compileGraph(&g, {"optimizer"}, {{"optimizer", std::string("sgd")}, {"learning_rate", 0.005f}});

  nnet::engine::NaiveEngine engine(g);
  engine.randomize();

  auto dataset = mnist::read_dataset_direct<std::vector, std::vector<uint8_t>>("./3rdparty/mnist/");

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
    nnet::eigen::cast<nnet::eigen::Matrix>(inputTensor).array() /= 255.0;
    engine.resetOrCreateGradient();
    engine.run(false);
    if (printGradMean) {
      engine.printMean();  // print mean grad of params
    }
    nnet::engine::Tensor lossMemTensor;
    lossMemTensor.buffer_ = nnet::memory::TensorBuffer::gTensorBuffers.at(avgLoss->name_);
    lossMemTensor.attr_ = avgLoss;
    auto m = nnet::eigen::cast<nnet::eigen::Vector>(lossMemTensor).array();
    LOG(INFO) << "MNIST batch-id="<< i <<" XE-Loss = " << *m.data();
  }
}

int main() {
  nnet::util::InitFunction::apply();
  bool runMNIST = true;
  if (runMNIST) {
    TrainMnistOnePass();
    nnet::memory::TensorBuffer::gTensorBuffers.clear(); // drop all buffer, make context clean.
  }

  return 0;
}