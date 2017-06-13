#include <Eigen/SparseCore>
#include <iostream>
// should use glog instead of elpp, because we could throw a Error when
// log(Fatal)
#include <easylogging++.h>
#include <mnist/mnist_reader.hpp>
#include "engine/Engine.h"
#include "graph/ComputationGraph.h"
#include "misc/CastEigen.h"
#include "misc/Error.h"
#include "misc/InitFunction.h"
#include "memory/Workspace.h"

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
  explicit inline GraphBuilder(memory::Workspace& workspace, graph::Graph* g) : graph_(g), workspace_(workspace) {}

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

  graph::TensorAttrPtr errorRate(const std::string& paramPrefix, graph::TensorAttrPtr prediction,
                                 graph::TensorAttrPtr label) {
    auto errorRate = graph_->createOrGetTensor(paramPrefix, {0}, false, graph::kTENSOR_FLOAT32);
    addOp("error_rate", {prediction, label}, {errorRate});
    return errorRate;
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

    auto paramTensor =
        graph_->createOrGetTensor(paramPrefix + ".param.weight.0", {layerWidth, size}, true, graph::kTENSOR_FLOAT32);
    workspace_(paramTensor);
    SmallVec<graph::TensorAttrPtr> inputs = {input, paramTensor, nullptr};
    if (withBias) {
      auto biasTensor = graph_->createOrGetTensor(paramPrefix + ".param.bias", {size, 1}, true, graph::kTENSOR_FLOAT32);
      workspace_(biasTensor);
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
  memory::Workspace& workspace_;
};
}
}

static void TrainMnistOnePass(size_t numPasses = 10, bool printGradMean = false) {
  nnet::graph::Graph g;
  constexpr size_t BATCH_SIZE = 1000;
  nnet::memory::Workspace w;

  nnet::api::GraphBuilder builder(w, &g);
  auto xTensor = g.createOrGetTensor("X", {BATCH_SIZE, 784}, false, nnet::graph::kTENSOR_FLOAT32);

  auto hidden = builder.fullyConnected("fc1", xTensor, 100, true);
  hidden = builder.fullyConnected("fc2", hidden, 100, true);
  auto prediction = builder.fullyConnected("prediction", hidden, 10, true, nnet::api::kSoftmax);
  auto labelTensor = g.createOrGetTensor("Label", {BATCH_SIZE, 1}, false, nnet::graph::kTENSOR_INT32);
  auto loss = builder.crossEntropy("xe_loss", prediction, labelTensor);
  auto errorRate = builder.errorRate("error_rate", prediction, labelTensor);
  auto avgLoss = builder.mean("avg_loss", loss);

  builder.backward(avgLoss);
  nnet::graph::compileGraph(&g, {"optimizer"}, {{"optimizer", std::string("sgd")}, {"learning_rate", 1.0f}});

  nnet::engine::NaiveEngine engine(w, g);
  engine.randomize();

  auto dataset = mnist::read_dataset_direct<std::vector, std::vector<uint8_t>>("./3rdparty/mnist/");
  for (size_t passId = 0; passId < numPasses; ++passId) {
    for (size_t i = 0; i < dataset.training_images.size() / BATCH_SIZE; ++i) {
      auto buf = (float*)w(xTensor)->get();
      auto labelBuf = (int*)w(labelTensor)->get();
      for (size_t j = 0; j < BATCH_SIZE; ++j) {
        auto& img = dataset.training_images[j + i * BATCH_SIZE];
        auto& lbl = dataset.training_labels[j + i * BATCH_SIZE];
        for (size_t k = 0; k < 784; ++k) {
          buf[j * 784 + k] = img[k];
        }
        labelBuf[j] = lbl;
      }
      nnet::eigen::cast<nnet::eigen::Matrix>(w.getTensor(xTensor)).array() /= 255.0;
      engine.resetOrCreateGradient();
      engine.run(false);
      if (printGradMean) {
        engine.printMean();  // print mean grad_ of params
      }

      auto avgLossArr = nnet::eigen::cast<nnet::eigen::Vector>(w.getTensor(avgLoss)).array();
      auto errRateArr = nnet::eigen::cast<nnet::eigen::Vector>(w.getTensor(errorRate)).array();
      LOG(INFO) << "MNIST pass-id=" << passId << " batch-id=" << i << " XE-Loss = " << *avgLossArr.data()
                << " error_rate = " << *errRateArr.data() * 100 << "%";
    }
  }
}

int main() {
  nnet::util::InitFunction::apply();
  bool runMNIST = false;
  if (runMNIST) {
    TrainMnistOnePass(10);
  }

  float A[]  = { 5, 8, 3, 6 };
  int IA[] = { 0, 0, 2, 3, 4 };
  int JA[] = { 0, 1, 2, 1 };

  Eigen::Map<Eigen::SparseMatrix<float, Eigen::RowMajor>> sm(4, 4, 4, IA, JA, A);

  LOG(INFO) << sm;

  return 0;
}
