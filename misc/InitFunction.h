#pragma once
#include <algorithm>
#include <functional>
#include "Error.h"
#include "Typedef.h"
namespace nnet {
namespace util {

class InitFunction final {
 public:
  using InitFN = std::function<void()>;
  InitFunction(InitFN fn, int pri = 0) { globalFuncs().push_back(std::make_pair(pri, fn)); }

  static void apply() {
    std::sort(globalFuncs().begin(), globalFuncs().end(),
              [](std::pair<int, InitFN>& a, std::pair<int, InitFN>& b) { return std::less<int>()(a.first, b.first); });

    for (auto& fn : globalFuncs()) {
      fn.second();
    }
  }

 private:
  static Vec<std::pair<int, InitFN>>& globalFuncs() {
    static Vec<std::pair<int, InitFN>> funcs;
    return funcs;
  };
};
}
}