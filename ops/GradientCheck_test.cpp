#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <engine/Engine.h>
#include <misc/CastEigen.h>
#include <catch.hpp>
#include <json.hpp>
#include <memory>
#include <random>
#include "misc/InitFunction.h"

static void randomize(const nnet::graph::Tensor& t) {
  auto m = nnet::eigen::cast<nnet::eigen::Matrix>(t);
  m.setRandom();
}

static void randomize(const nnet::graph::Tensor& t, int min_value, int max_value) {
  int* buf = (int*)t.buffer_->get();
  std::random_device dev;
  std::mt19937 engine(dev());
  std::uniform_int_distribution<int> dist(min_value, max_value);
  for (size_t i=0; i< nnet::details::product(t.attr_->dims_); ++i) {
    buf[i] = dist(engine);
  }
}

static void randomizeProb(const nnet::graph::Tensor& t) {
  float* buf = (float*)t.buffer_->get();
  std::random_device dev;
  std::mt19937 engine(dev());
  std::uniform_real_distribution<float> dist(0.0f, 2.0f);
  for (size_t r=0; r<t.attr_->dims_[0]; ++r) {
    float rowSum = 0.0;
    for (size_t c=0; c<t.attr_->dims_[1]; ++c) {
      buf[r*t.attr_->dims_[1] + c] = dist(engine);
      rowSum += buf[r*t.attr_->dims_[1] + c];
    }
    for (size_t c=0; c<t.attr_->dims_[1]; ++c) {
      buf[r*t.attr_->dims_[1] + c] /= rowSum;
    }
  }
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
    "type": "softmax",
    "inputs": [
      ["input", [20, 10], true, "f", true]
    ],
    "output": ["output", [20, 10], true, "f", false]
  },
  {
    "type": "cross_entropy",
    "inputs": [
      ["prob", [20, 10], true, "f", true, {
        "is_prob": true
      }],
      ["label", [20, 1], false, "i", false, {
        "max_value": 9,
        "min_value": 0
      }]
    ],
    "output": ["output", [20, 1], true, "f", false]
  },
  {
    "type": "sigmoid",
    "inputs": [
      ["sigmoid.param", [10, 20], true, "f", true]
    ],
    "output": ["output", [10, 20], true, "f", false]
  },
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
      ["input", [100, 20], false, "f", false],
      ["fc.param", [20, 10], true, "f", true],
      []
    ],
    "output": ["output", [100, 10], true, "f", false]
  }
]
)"_json;
  size_t testId = 0;
  for (nlohmann::json& metaInfo : metaInfos) {
    std::string opType = metaInfo["type"];
    nnet::graph::Graph graph;
    nnet::SmallVec<nnet::graph::TensorAttrPtr> input;
    nlohmann::json& inputJson = metaInfo["inputs"];
    nnet::memory::Workspace workspace;

    size_t gcPoint = -1UL;
    for (nlohmann::json& eachInput : inputJson) {
      input.push_back(toAttr(&graph, eachInput));
      if (gcPoint == -1UL) {
        bool isGC = eachInput[4];
        if (isGC) {
          gcPoint = input.size() - 1;
        }
      }
      auto & attr = input.back();
      if (attr == nullptr) continue;
      if (eachInput.size() == 5UL) {
        randomize(workspace.getTensor(attr));
      } else {
        nlohmann::json & randomOption = eachInput[5];
        if (attr->type_ == nnet::graph::kTENSOR_INT32) {
          int min = randomOption["min_value"];
          int max = randomOption["max_value"];
          randomize(workspace.getTensor(attr), min, max);
        } else {
          auto it = randomOption.find("is_prob");
          if (it != randomOption.end()) {
            bool is_prob = randomOption["is_prob"];
            randomizeProb(workspace.getTensor(attr));
          } else {
            LOG(FATAL) << "Not implement";
          }
        }
      }
    }
    REQUIRE(gcPoint != -1UL);
    nnet::graph::TensorAttrPtr output = toAttr(&graph, metaInfo["output"]);
    LOG(INFO) << "Gradient check op " << opType << ", input index=" << gcPoint;
    auto gcTensor = workspace.getTensor(input[gcPoint]);
    graph.ops_.push_back(nnet::graph::Op(opType, input, {output}));
    auto meanOut = graph.createOrGetTensor("mean", {1, 1}, true, nnet::graph::kTENSOR_FLOAT32);
    graph.ops_.push_back(nnet::graph::Op("mean", {output}, {meanOut}));
    nnet::eigen::Matrix wGrad(gcTensor.attr_->dims_[0], gcTensor.attr_->dims_[1]);
    auto mW = nnet::eigen::cast<nnet::eigen::Matrix>(gcTensor);
    {
      LOG(INFO) << "Generating gradient checking";
      nnet::engine::NaiveEngine engine(workspace, graph);
      for (size_t h = 0; h < wGrad.rows(); ++h) {
        for (size_t w = 0; w < wGrad.cols(); ++w) {
          mW(h, w) += epsilon;
          engine.run();
          auto m = workspace.getTensor(meanOut);
          float hi = *(float*)m.buffer_->get();
          mW(h, w) -= 2 * epsilon;
          engine.run();
          float low = *(float*)m.buffer_->get();
          mW(h, w) += epsilon;
          wGrad(h, w) = (hi - low) / (2 * epsilon);
        }
      }
    }
    {
      LOG(INFO) << "Backwarding";
      nnet::Map<std::string, nnet::Any> attrs;
      attrs.insert({"loss_name", meanOut->name_});
      nnet::graph::compileGraph(&graph, {"backward"}, attrs);
      nnet::engine::NaiveEngine engine(workspace, graph);
      engine.resetOrCreateGradient();
      auto wg = workspace.getTensor(graph.tensors_.at(gcTensor.attr_->name_ + ".grad"));
      auto mWG = nnet::eigen::cast<nnet::eigen::Matrix>(wg);
      engine.run();
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
    printf("\n");
  }
}