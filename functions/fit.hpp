#pragma once

#include "../perceptron/perceptron.hpp"
#include "../util/fit_util.hpp"
#include <iostream>
#include <random>

namespace fit {

bool random_fit(Perceptron::WeightVector *weights, Perceptron *perceptron,
                const Perceptron::transformed_training_data &training_set,
                const double &learning_rate, const ulong &epochs,
                ulong &iterations, const double &accuracy) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  while (!almost_equal(perceptron->predict(training_set.first),
                       training_set.second, accuracy)) {
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

bool adaline_fit_noreaval(
    Perceptron::WeightVector *weights, Perceptron *perceptron,
    const Perceptron::transformed_training_data &training_set,
    const double &learning_rate, const ulong &epochs, ulong &iterations,
    const double &epsilon) {

  std::vector<double> output = perceptron->predict(training_set.first, false);
  auto input_it = training_set.first.begin();
  auto res_it = output.begin();
  auto actual_it = training_set.second.begin();

  while (!almost_equal(output, training_set.second, epsilon)) {
    auto weights_it = weights->begin();
    for (int i = -1; i < input_it->size(); ++i) {
      double input_value = 1;

      if (i == input_it->size()) {
        i = -1;
      }

      if (i >= 0) {
        input_value = input_it[0][i];
        weights_it = weights->begin() + i + 1;
      }

      double difference = *actual_it - *res_it;
      // Change weights according to ADALINE rule
      *weights_it += learning_rate * difference * input_value;
    }

    ++input_it;
    ++actual_it;
    ++res_it;
    if (input_it == training_set.first.end()) {
      input_it = training_set.first.begin();
      res_it = output.begin(), actual_it = training_set.second.begin();
      auto new_output = perceptron->predict(training_set.first, false);
      output.assign(new_output.begin(), new_output.end());
    }
  }
  return true;
}

bool adaline_fit(Perceptron::WeightVector *weights, Perceptron *perceptron,
                 const Perceptron::transformed_training_data &training_set,
                 const double &learning_rate, const ulong &epochs,
                 ulong &iterations, const double &epsilon) {

  std::vector<double> results = perceptron->predict(training_set.first, false);
  auto input_it = training_set.first.begin();
  auto res_it = results.begin();
  auto actual_it = training_set.second.begin();

  while (!almost_equal(results, training_set.second, epsilon)) {
    auto weights_it = weights->begin();
    for (int i = -1; i < input_it->size(); ++i) {
      double input_value = 1;

      if (i == input_it->size()) {
        i = -1;
      }

      if (i >= 0) {
        input_value = input_it[0][i];
        weights_it = weights->begin() + i + 1;
      }

      double difference = *actual_it - *res_it;
      // Change weights according to ADALINE rule
      *weights_it += learning_rate * difference * input_value;
    }
    *res_it = perceptron->predict({*input_it}, false)[0];
    if (*res_it != *actual_it) {
      continue;
    }

    ++input_it;
    ++actual_it;
    ++res_it;
    if (input_it == training_set.first.end()) {
      input_it = training_set.first.begin();
      res_it = results.begin(), actual_it = training_set.second.begin();
    }
    auto new_output = perceptron->predict(training_set.first, false);
    results.assign(new_output.begin(), new_output.end());
  }
  return true;
}

} // namespace fit
