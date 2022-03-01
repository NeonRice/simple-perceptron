#include "perceptron.hpp"
#include <climits>
#include <cmath>

std::vector<int>
Perceptron::predict(const std::vector<std::array<double, 2>> &input) {
  std::vector<int> results;
  results.reserve(input.size());
  auto weight_it = this->weights.cbegin();
  auto input_pair_it = input.cbegin();

  while (input_pair_it != input.cend()) {
    double weighted_sum = 0;
    // w0 - Bias
    double bias = *(weight_it++);
    for (auto input_it = input_pair_it->cbegin();
         input_it != input_pair_it->cend() && weight_it != weights.cend();
         ++input_it, ++weight_it) {
      weighted_sum += (*weight_it) * (*input_it);
    }
    ++input_pair_it;
    weight_it = this->weights.cbegin();
    results.push_back(round(activation_function(weighted_sum + bias)));
  }
  return results;
}

std::pair<bool, ulong>
Perceptron::train(const std::vector<std::array<double, 2>> &input,
                  const std::vector<int> &output, const ulong &epochs,
                  const double &learning_rate) {
  ulong iterations = 0;
  this->initialization_function(&weights);
  while (!fitting_function(&weights, this, input, output, learning_rate, epochs,
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
