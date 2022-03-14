#pragma once

#include "perceptron.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

class DynamicPerceptron
    : public Perceptron<Eigen::Matrix<double, 1, Eigen::Dynamic>,
                        Eigen::Matrix<double, 1, Eigen::Dynamic>,
                        DynamicPerceptron> {
public:
  typedef Eigen::Matrix<double, 1, Eigen::Dynamic> WeightVector;
  using InputVector = WeightVector;
  DynamicPerceptron(init_fun init, active_fun activate, fit_fun fit)
      : Perceptron(init, activate, fit) {}

   // DynamicPerceptron(init_fun init, active_fun active, fit_fun fit)
   //     : initialization_function(init), activation_function(active),
   //       fitting_function(fit) {}

    std::pair<bool, ulong> train(const training_data &training_set,
                                 const ulong &epochs = 250,
                                 const double &learning_rate = 0.1) {
      ulong iterations = 0;
      this->initialization_function(&weights);
      while (!fitting_function(&weights, this, training_set, learning_rate,
                               epochs, iterations)) {
        if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
          // Training failed, not enough epochs?
          // If epochs = 0, never stop until weights found
          return std::pair<bool, ulong>(false, iterations);
        }
      }
      return std::pair<bool, ulong>(true, iterations);
    }
    std::vector<int> predict(const std::vector<InputVector> &input) {
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
        WeightVector input;
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
    inline void set_fit_function(fit_fun fun) { this->fitting_function = fun; }

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
