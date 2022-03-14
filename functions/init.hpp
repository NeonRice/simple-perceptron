#pragma once

#include "../perceptron/perceptron.hpp"
#include "../perceptron/dynamic_perceptron.hpp"
#include <cmath>
#include <random>

namespace init {

template <typename Container>
void initialize_weights(Container *weights) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

} // namespace init
