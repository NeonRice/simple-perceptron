#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace util {

double calculate_error(const std::vector<double> &a,
                       const std::vector<int> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Training set pairs must be of equal size");
  }
  double difference_sum = 0;
  for (int i = 0; i < a.size(); ++i) {
    difference_sum += pow(b[i] - a[i], 2);
  }

  return 0.5 * difference_sum;
}

bool almost_equal(const std::vector<double> &a, const std::vector<int> &b,
                  const double &accuracy = 0.2) {
  if (calculate_error(a, b) > accuracy) {
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

} // namespace util
