#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <catch.hpp>
#include <misc/CastEigen.h>
#include <engine/Engine.h>
#include "graph/ComputationGraph.h"
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
  return nnet::graph::Tensor  {ptr, buffer};
}

static void randomize(const nnet::graph::Tensor& t) {
  auto m = nnet::eigen::cast<nnet::eigen::Matrix >(t);
  m.setRandom();
}

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

static void printProgress (double percentage)
{
  int val = (int) (percentage * 100);
  int lpad = (int) (percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush (stdout);
}


TEST_CASE("GradientCheck", "setup") {
  nnet::util::InitFunction::apply();
  constexpr float epsilon = 1e-4;

  SECTION("GradientCheck", "fc") {
    nnet::graph::Graph graph;
    size_t BatchSize = 100;
    size_t InSize = 200;
    size_t FcSize = 50;

    auto inAttr = graph.createOrGetTensor("input", {BatchSize, InSize}, false, nnet::graph::kTENSOR_FLOAT32);
    auto wAttr = graph.createOrGetTensor("fc.param", {InSize, FcSize}, true, nnet::graph::kTENSOR_FLOAT32);
    auto oAttr = graph.createOrGetTensor("output", {BatchSize, FcSize}, true, nnet::graph::kTENSOR_FLOAT32);
    graph.ops_.push_back(nnet::graph::Op("fc", {inAttr, wAttr, nullptr}, {oAttr}));

    auto meanOut = graph.createOrGetTensor("mean", {1, 1}, true, nnet::graph::kTENSOR_FLOAT32);
    graph.ops_.push_back(nnet::graph::Op("mean", {oAttr}, {meanOut}));


    auto in = toTensor(inAttr);
    auto w = toTensor(wAttr);
    auto o = toTensor(oAttr);
    auto m = toTensor(meanOut);

    randomize(in);
    randomize(o);
    randomize(w);

    nnet::eigen::Matrix wGrad(w.attr_->dims_[0], w.attr_->dims_[1]);
    auto mW = nnet::eigen::cast<nnet::eigen::Matrix >(w);
    {
      nnet::engine::NaiveEngine engine(graph);

      for (size_t h = 0; h < wGrad.rows(); ++h) {
        for (size_t w = 0; w < wGrad.cols(); ++w) {
          mW(h, w) += epsilon;
          engine.run();
          float hi = *(float *) m.buffer_->get();
          mW(h, w) -= 2 * epsilon;
          engine.run();
          float low = *(float *) m.buffer_->get();
          mW(h, w) += epsilon;
          wGrad(h, w) = (hi - low) / (2 * epsilon);
          printProgress(((double) (h * wGrad.cols() + w)) / (wGrad.cols() * wGrad.rows()));
        }
      }
    }
    {
      nnet::Map<std::string, nnet::Any> attrs;
      attrs.insert({"loss_name", meanOut->name_});
      nnet::graph::compileGraph(&graph, {"backward"}, attrs);
      nnet::engine::NaiveEngine engine(graph);
      engine.resetOrCreateGradient();
      engine.run();
      auto wg = graph.getTensor("fc.param.grad");
      auto mWG = nnet::eigen::cast<nnet::eigen::Matrix >(wg);


      nnet::eigen::Matrix lower(w.attr_->dims_[0], w.attr_->dims_[1]);
      nnet::eigen::Matrix upper(w.attr_->dims_[0], w.attr_->dims_[1]);
      upper.array() = wGrad.array() - mWG.array();
      upper.array() = upper.array().abs();

      lower = wGrad;
      lower.array() += epsilon;
      wGrad.array() = upper.array() / lower.array();
    }

    LOG(INFO) << "Diff = " << wGrad.mean();
  }
}