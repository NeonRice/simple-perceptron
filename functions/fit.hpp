#pragma once

//#define DEBUG

#include "../perceptron/perceptron.hpp"
#include "../util/fit_util.hpp"
#include <iostream>
#include <random>
#include <climits>

using namespace util;

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

bool adaline_fit(Perceptron::WeightVector *weights, Perceptron *perceptron,
                 const Perceptron::transformed_training_data &training_set,
                 const double &learning_rate, const ulong &epoch_cnt,
                 ulong &epochs, const double &epsilon) {

  std::vector<double> results = perceptron->predict(training_set.first, false);
  std::pair<double, Perceptron::WeightVector> best_results = {INT_MAX, *weights};
  auto input_it = training_set.first.begin();
  auto res_it = results.begin();
  auto actual_it = training_set.second.begin();

  // if epoch_cnt == 0, means loop until solution found
  while (epoch_cnt == 0 || epochs < epoch_cnt) {
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
    if (!almost_equal({*res_it}, {*actual_it}, epsilon)) {
      continue;
    }

    ++input_it;
    ++actual_it;
    ++res_it;
    if (input_it == training_set.first.end()) {
      input_it = training_set.first.begin();
      res_it = results.begin(), actual_it = training_set.second.begin();

      auto new_output = perceptron->predict(training_set.first, false);
      results.assign(new_output.begin(), new_output.end());

      double error = calculate_error(results, training_set.second);
#ifdef DEBUG
      std::cout << "Epoch error: " << error << std::endl;
#endif

      if (error < best_results.first) {
        best_results.first = error;
        best_results.second = *weights;
      }

      ++epochs;
      if (almost_equal(results, training_set.second, epsilon)) {
        break;
      }
    }
  }

  if (best_results.second != *weights) {
    *weights = best_results.second;
  }

  return true;
}

} // namespace fit
