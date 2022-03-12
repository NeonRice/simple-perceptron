#include "perceptron.hpp"
#include <climits>
#include <cmath>
#include <string>

using namespace Eigen;

std::vector<int>
Perceptron::predict(const std::vector<VectorXd> &input) {
  std::vector<int> results;
  results.reserve(input.size());
  auto input_pair_it = input.cbegin();

  while (input_pair_it != input.cend()) {
    if (input_pair_it->size() + 1 != this->weights.size()) {
      throw "Input vector size is " + std::to_string(input_pair_it->size()) +
          "expected size " + std::to_string(this->weights.size() - 1);
    }
    VectorXd input(input_pair_it->size() + 1);
    input << 1, *input_pair_it;
    double weighted_sum = input_pair_it->dot(this->weights.transpose());
    results.push_back(round(activation_function(weighted_sum)));
  }
  return results;
}

std::pair<bool, ulong> Perceptron::train(const training_data &training_set,
                                         const ulong &epochs,
                                         const double &learning_rate) {
  ulong iterations = 0;
  this->initialization_function(&weights);
  while (!fitting_function(&weights, this, training_set, learning_rate, epochs,
                           iterations)) {
    if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
      // Training failed, not enough epochs?
      // If epochs = 0, never stop until weights found
      return std::pair<bool, ulong>(false, iterations);
    }
  }
  return std::pair<bool, ulong>(true, iterations);
}

Perceptron::Perceptron(init_fun init, active_fun active, fit_fun fit)
    : initialization_function(init), activation_function(active),
      fitting_function(fit) {}
