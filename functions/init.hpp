#pragma once

#include "../perceptron/perceptron.hpp"
#include "../perceptron/dynamic_perceptron.hpp"
#include <cmath>
#include <random>

namespace init {

template <typename WeightVector>
void initialize_weights(WeightVector *weights) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

// TODO: Refactor to use generic class
void initialize_weights(DynamicPerceptron::WeightVector *weights) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

// TODO: Refactor to use generic class
void initialize_weights(DynamicPerceptron::WeightVector *weights) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

} // namespace init
