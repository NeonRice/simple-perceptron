#pragma once

#include <Eigen/Dense>
#include <array>
#include <vector>

class Perceptron {
public:
  typedef Eigen::Matrix<double, 1, Eigen::Dynamic> WeightVector;
  typedef std::vector<std::pair<WeightVector, int>> training_data;
  typedef std::pair<std::vector<WeightVector>, std::vector<int>>
      transformed_training_data;

  using init_fun = void (*)(WeightVector *weights);

  using active_fun = double (*)(const double &weighted_sum);

  using fit_fun = bool (*)(WeightVector *weights, Perceptron *perceptron,
                           const transformed_training_data &training_set,
                           const double &learning_rate, const ulong &epochs,
                           ulong &iterations);

  Perceptron(init_fun init, active_fun active, fit_fun fit)
      : initialization_function(init), activation_function(active),
        fitting_function(fit) {}
  ~Perceptron() = default;

  // Get transformed_training_data from training_data
  // Good for use with predict function
  inline static transformed_training_data
  transform_training_data(const training_data &data) {
    std::pair<std::vector<WeightVector>, std::vector<int>> transformed_set;
    for (auto training_sample : data) {
      transformed_set.first.push_back(training_sample.first);
      transformed_set.second.push_back(training_sample.second);
    }
    return transformed_set;
  }

  // Get training_data from transformed_training_data
  inline static training_data
  transform_training_data(const transformed_training_data &data) {
    training_data training_set;
    if (data.first.size() != data.second.size()) {
      throw std::invalid_argument(
          "Transformed training data vector sizes must be equal");
    }
    for (uint i = 0; i < data.second.size(); ++i) {
      training_set.push_back(
          std::pair<WeightVector, int>(data.first[i], data.second[i]));
    }
    return training_set;
  }

  std::pair<bool, ulong> train(const training_data &training_set,
                                       const ulong &epochs = 250,
                                       const double &learning_rate = 0.1) {
    ulong iterations = 0;
    weights = WeightVector(training_set[0].first.size() + 1);
    this->initialization_function(&weights);
    while (!fitting_function(&weights, this,
                             transform_training_data(training_set),
                             learning_rate, epochs, iterations)) {
      if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
        // Training failed, not enough epochs?
        // If epochs = 0, never stop until weights found
        return std::pair<bool, ulong>(false, iterations);
      }
    }
    return std::pair<bool, ulong>(true, iterations);
  }

  std::vector<int> predict(const std::vector<WeightVector> &input) {
    std::vector<int> results;
    results.reserve(input.size());
    auto input_pair_it = input.cbegin();

    while (input_pair_it != input.cend()) {
      if (input_pair_it->size() != this->weights.size() - 1) {
        throw std::invalid_argument(
            "Input vector size is " + std::to_string(input_pair_it->size()) +
            " expected size " + std::to_string(this->weights.size() - 1));
      }
      // WeightVector because we need x0 to be 1 (for bias)
      // or in other words k + 1 members
      WeightVector input = WeightVector(input_pair_it->size() + 1);
      input << 1, *input_pair_it++;
      double weighted_sum = input.dot(this->weights.transpose());
      results.push_back(round(activation_function(weighted_sum)));
    }
    return results;
  }

  inline void set_activation_function(active_fun fun) {
    this->activation_function = fun;
  }
  inline void set_init_function(init_fun fun) {
    this->initialization_function = fun;
  }
  inline void set_fit_function(fit_fun fun) {
    this->fitting_function = fun;
  }

  inline WeightVector get_weights() { return weights; }
  inline WeightVector *set_weights(const WeightVector &weights) {
    this->weights = weights;
    return &this->weights;
  }

private:
  WeightVector weights;
  double weighted_sum = 0;

  // Functions
  init_fun initialization_function;
  active_fun activation_function;
  fit_fun fitting_function;
};
