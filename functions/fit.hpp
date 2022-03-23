#pragma once

#include "../perceptron/perceptron.hpp"
#include <iostream>
#include <random>

namespace fit {

// bool random_fit(Perceptron::WeightVector *weights, Perceptron *perceptron,
//                const Perceptron::transformed_training_data &training_set,
//                const double &learning_rate, const ulong &epochs,
//                ulong &iterations) {
//  std::mt19937 gen(time(NULL));
//  std::uniform_real_distribution<double> dist(-3, 3);
//  while (perceptron->predict(training_set.first) != training_set.second) {
//    if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
//      // Training failed, not enough epochs?
//      // If epochs = 0, never stop until weights found
//      return false;
//    }
//    for (double &weight : *weights) {
//      weight = dist(gen);
//    }
//  }
//  return true;
//}

bool almost_equal(const std::vector<double> &a, const std::vector<int> &b,
                  const double &accuracy = 0.2) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Training set pairs must be of equal size");
  }
  double difference_sum = 0;
  for (int i = 0; i < a.size(); ++i) {
    difference_sum += pow(b[i] - a[i], 2);
  }
  auto error = 0.5 * difference_sum;
  if (error > accuracy) {
    std::cout << error << std::endl;
    return false;
  }

  return true;
}

bool almost_equal_epsilon(std::vector<double> a, std::vector<int> b,
                          double epsilon = 0.2) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Training set pairs must be of equal size");
  }
  for (int i = 0; i < a.size(); ++i) {
    if (fabs(b[i] - a[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

bool adaline_fit(Perceptron::WeightVector *weights, Perceptron *perceptron,
                 const Perceptron::transformed_training_data &training_set,
                 const double &learning_rate, const ulong &epochs,
                 ulong &iterations) {

  constexpr double epsilon = 0.2;
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
      // for (auto weights_it = weights->begin(); weights_it != weights->end();
      //     ++weights_it) {
      //  std::cout << *weights_it << " ";
      //}
      // std::cout << std::endl;
      // if (weights_it == weights->end()) {
      //  weights_it = weights->begin();
      //}
    }
    *res_it = perceptron->predict({*input_it}, false)[0];

    ++input_it;
    ++actual_it;
    ++res_it;
    if (input_it == training_set.first.end()) {
      input_it = training_set.first.begin();
      res_it = output.begin(), actual_it = training_set.second.begin();
    }
    //auto new_output = perceptron->predict(training_set.first, false);
    //output.assign(new_output.begin(), new_output.end());
  }
  return true;
}

} // namespace fit
