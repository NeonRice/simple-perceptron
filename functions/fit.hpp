#pragma once

#include "../perceptron/perceptron.hpp"
#include <random>

namespace fit {

template <unsigned int k>
bool random_fit(typename Perceptron<k>::WeightVector *weights, Perceptron<k> *perceptron,
                const typename Perceptron<k>::training_data &training_set,
                const double &learning_rate, const ulong &epochs,
                ulong &iterations) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  while (perceptron->predict(training_set.first) != training_set.second) {
    if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
      // Training failed, not enough epochs?
      // If epochs = 0, never stop until weights found
      return false;
    }
    for (double &weight : *weights) {
      weight = dist(gen);
    }
  }
  return true;
}

} // namespace fit
