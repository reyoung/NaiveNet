//
// Created by baidu on 2017/6/5.
//

#include "ComputationGraph.h"

namespace nnet {
namespace graph {
Map<std::string, OpMeta> OpMeta::gAllOpMeta_;

extern Map<std::string, CompileGraphFN>& compilers() {
  static Map<std::string, CompileGraphFN> gCompilers;
  return gCompilers;
};
}
}