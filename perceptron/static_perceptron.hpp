#pragma once

#include "perceptron.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

template <unsigned int k>
class StaticPerceptron
    : public Perceptron<Eigen::Matrix<double, 1, k + 1>, Eigen::Matrix<double, 1, k>> {
public:
  typedef Eigen::Matrix<double, 1, k + 1> WeightVector;
  typedef Eigen::Matrix<double, 1, k> InputVector;
  typedef std::pair<std::vector<InputVector>, std::vector<int>> training_data;
};
