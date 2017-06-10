#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <engine/Engine.h>
#include <misc/CastEigen.h>
#include <catch.hpp>
#include <json.hpp>
#include <memory>
#include "misc/InitFunction.h"

static nnet::graph::Tensor toTensor(const nnet::graph::TensorAttrPtr& ptr) {
  std::shared_ptr<nnet::memory::TensorBuffer> buffer;
  if (ptr->type_ == nnet::graph::kTENSOR_FLOAT32) {
    buffer = nnet::memory::TensorBuffer::createOrResizeBuffer<float>(ptr->name_, ptr->dims_);
  } else if (ptr->type_ == nnet::graph::kTENSOR_INT32) {
    buffer = nnet::memory::TensorBuffer::createOrResizeBuffer<int>(ptr->name_, ptr->dims_);
  } else {
    REQUIRE(false);
  }
  return nnet::graph::Tensor{ptr, buffer};
}

static void randomize(const nnet::graph::Tensor& t) {
  auto m = nnet::eigen::cast<nnet::eigen::Matrix>(t);
  m.setRandom();
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

static void printProgress(double percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

static nnet::SmallVec<size_t> toDims(const nlohmann::json& a) {
  nnet::SmallVec<size_t> dims;
  for (const nlohmann::json& item : a) {
    dims.push_back((size_t)item);
  }
  return dims;
}

static nnet::graph::TensorAttrPtr toAttr(nnet::graph::Graph* g, const nlohmann::json& j) {
  if (j.size() == 0) {
    return nullptr;
  }
  nnet::graph::TensorType type;
  if (j[3] == "f") {
    type = nnet::graph::kTENSOR_FLOAT32;
  } else if (j[3] == "i") {
    type = nnet::graph::kTENSOR_INT32;
  } else {
    LOG(FATAL) << "unexpected branch";
  }
  return g->createOrGetTensor(j[0], toDims(j[1]), j[2], type);
}

TEST_CASE("GradientCheck", "check_all") {
  nnet::util::InitFunction::apply();
  constexpr float epsilon = 1e-3;
  auto F = nnet::graph::kTENSOR_FLOAT32;

  nlohmann::json metaInfos = R"([
  {
    "type": "fc",
    "inputs": [
      ["input", [100, 200], false, "f", false],
      ["fc.param", [200, 50], true, "f", false],
      ["fc.bias", [1, 50], true, "f", true]
    ],
    "output": ["output", [100, 50], true, "f", false]
  },
  {
    "type": "fc",
    "inputs": [
      ["input", [100, 200], false, "f", false],
      ["fc.param", [200, 50], true, "f", true],
      []
    ],
    "output": ["output", [100, 50], true, "f", false]
  }
]
)"_json;
  size_t testId = 0;
  for (nlohmann::json& metaInfo : metaInfos) {
    std::string opType = metaInfo["type"];
    nnet::graph::Graph graph;
    nnet::SmallVec<nnet::graph::TensorAttrPtr> input;
    nlohmann::json& inputJson = metaInfo["inputs"];

    size_t gcPoint = -1UL;
    for (nlohmann::json& eachInput : inputJson) {
      input.push_back(toAttr(&graph, eachInput));
      if (gcPoint == -1UL) {
        bool isGC = eachInput[4];
        if (isGC) {
          gcPoint = input.size() - 1;
        }
      }
    }
    REQUIRE(gcPoint != -1UL);
    nnet::graph::TensorAttrPtr output = toAttr(&graph, metaInfo["output"]);
    LOG(INFO) << "Gradient check op " << opType << ", input index=" << gcPoint;
    // randomize all input
    for (auto& attr : input) {
      if (attr == nullptr) continue;
      randomize(toTensor(attr));
    }
    auto gcTensor = toTensor(input[gcPoint]);
    graph.ops_.push_back(nnet::graph::Op(opType, input, {output}));
    auto meanOut = graph.createOrGetTensor("mean", {1, 1}, true, nnet::graph::kTENSOR_FLOAT32);
    graph.ops_.push_back(nnet::graph::Op("mean", {output}, {meanOut}));
    nnet::eigen::Matrix wGrad(gcTensor.attr_->dims_[0], gcTensor.attr_->dims_[1]);
    auto mW = nnet::eigen::cast<nnet::eigen::Matrix>(gcTensor);
    auto m = toTensor(meanOut);
    {
      LOG(INFO) << "Generating gradient checking";
      nnet::engine::NaiveEngine engine(graph);
      for (size_t h = 0; h < wGrad.rows(); ++h) {
        for (size_t w = 0; w < wGrad.cols(); ++w) {
          mW(h, w) += epsilon;
          engine.run();
          float hi = *(float*)m.buffer_->get();
          mW(h, w) -= 2 * epsilon;
          engine.run();
          float low = *(float*)m.buffer_->get();
          mW(h, w) += epsilon;
          wGrad(h, w) = (hi - low) / (2 * epsilon);
          printProgress(((double)(h * wGrad.cols() + w)) / (wGrad.cols() * wGrad.rows()));
        }
      }
      printf("\n");
      LOG(INFO) << "Done.";
    }
    {
      LOG(INFO) << "Backwarding";
      nnet::Map<std::string, nnet::Any> attrs;
      attrs.insert({"loss_name", meanOut->name_});
      nnet::graph::compileGraph(&graph, {"backward"}, attrs);
      nnet::engine::NaiveEngine engine(graph);
      engine.resetOrCreateGradient();
      engine.run();

      auto wg = graph.getTensor(gcTensor.attr_->name_ + ".grad");
      auto mWG = nnet::eigen::cast<nnet::eigen::Matrix>(wg);
      nnet::eigen::Matrix lower(gcTensor.attr_->dims_[0], gcTensor.attr_->dims_[1]);
      nnet::eigen::Matrix upper(gcTensor.attr_->dims_[0], gcTensor.attr_->dims_[1]);
      upper.array() = wGrad.array() - mWG.array();
      upper.array() = upper.array().abs();
      lower = wGrad;
      lower.array() += epsilon;
      wGrad.array() = upper.array() / lower.array();
      LOG(INFO) << "Done.";
    }
    REQUIRE(wGrad.mean() < 0.1f);
    ++testId;
    nnet::memory::TensorBuffer::gTensorBuffers.clear();  // clear all buffer before.
  }
}