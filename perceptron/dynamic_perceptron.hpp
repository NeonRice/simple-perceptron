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
};
