#pragma once
#include "engine/Engine.h"
#include "misc/CastEigen.h"
#include "misc/InitFunction.h"

namespace nnet {
namespace eigen_ops {
using eigen::cast;
using eigen::Matrix;
using eigen::Vector;
using eigen::IVector;
using util::InitFunction;
using graph::Variable;
using graph::VariableAttr;
using graph::VariableAttrPtr;
using graph::Op;
using graph::OpMeta;
using graph::AttributeMeta;
using graph::kDEVICE_CPU;
}
}