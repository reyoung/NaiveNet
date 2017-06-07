#pragma once
#include <boost/any.hpp>
#include <boost/container/small_vector.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nnet {

template <typename T1, typename T2>
using Map = std::unordered_map<T1, T2>;
template <typename T1>
using Set = std::unordered_set<T1>;
template <typename T1, size_t N>
using SmallVecN = boost::container::small_vector<T1, N>;

template <typename T1>
using SmallVec = SmallVecN<T1, 5>;

template <typename T>
bool operator==(const SmallVec<T>& vec1, const SmallVec<T>& vec2) {
  if (vec1.size() != vec2.size()) {
    return false;
  }
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec1[i] != vec2[i]) {
      return false;
    }
  }
  return true;
}

template <typename T1>
using Vec = std::vector<T1>;

using Any = boost::any;
using boost::any_cast;
using boost::bad_any_cast;

namespace details {

template <typename Container>
inline size_t product(Container c) {
  size_t prod = 1;
  for (auto& item : c) {
    prod *= item;
  }
  return prod;
}
}
}