#pragma once

#include "Eigen/Dense"
#include "types.hpp"
#include <iostream>
#include <vector>
#include <cmath>

namespace util {
template <class Iterable>
std::pair<Iterable, Iterable>
split(Iterable iterable, double split_by) {
  std::pair<Iterable, Iterable> result;

  if (split_by > 1 || split_by < 0) {
    throw std::invalid_argument("Split by must be in range [0;1]");
  }

  size_t size = iterable.end() - iterable.begin();
  size_t part1_sz = ceil(size * split_by);
  
  auto part1_end = iterable.begin() + part1_sz;

  bool part1_done = false;
  for (auto start_it = iterable.begin(); start_it != iterable.end(); ++start_it) {
    if (!part1_done && start_it != part1_end) {
      result.first.push_back(*start_it);
      continue;
    }
    part1_done = true;
    result.second.push_back(*start_it);
  }

  return result;
}

} // namespace util
