#pragma once

#include "perceptron.hpp"
#include <Eigen/Dense>
#include <array>
#include <vector>

template <unsigned int k>
class StaticPerceptron
    : public Perceptron<Eigen::Matrix<double, 1, k + 1>,
                        Eigen::Matrix<double, 1, k>, StaticPerceptron<k>> {
public:
  typedef Eigen::Matrix<double, 1, k + 1> WeightVector;
  typedef Eigen::Matrix<double, 1, k> InputVector;

  using init_fun = typename Perceptron<Eigen::Matrix<double, 1, k + 1>,
                                       Eigen::Matrix<double, 1, k>,
                                       StaticPerceptron<k>>::init_fun;

  using active_fun = typename Perceptron<Eigen::Matrix<double, 1, k + 1>,
                                         Eigen::Matrix<double, 1, k>,
                                         StaticPerceptron<k>>::active_fun;

  using fit_fun = typename Perceptron<Eigen::Matrix<double, 1, k + 1>,
                                      Eigen::Matrix<double, 1, k>,
                                      StaticPerceptron<k>>::fit_fun;

  StaticPerceptron(init_fun init, active_fun active, fit_fun fit)
      : Perceptron<Eigen::Matrix<double, 1, k + 1>, Eigen::Matrix<double, 1, k>,
                   StaticPerceptron<k>>(init, active, fit){};
};
