#pragma once

#include "../perceptron/perceptron.hpp"
#include <cmath>
#include <random>

namespace init {

template <typename WeightVector>
void initialize_weights(WeightVector *weights, const double& seed) {
  std::mt19937 gen(time(NULL) + seed);
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

} // namespace init
