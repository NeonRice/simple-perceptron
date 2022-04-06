#pragma once

#include "Eigen/Dense"
#include <iostream>
#include <vector>

namespace util {

typedef std::vector<std::pair<Eigen::Matrix<double, 1, Eigen::Dynamic>, int>>
    training_data;

} // namespace util
