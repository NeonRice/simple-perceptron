#pragma once

#include <cmath>

namespace activate {

// Fast sigmoid function
double fast_sigmoid(const double &x) { return x / (1 + fabs(x)); }

// Classical sigmoid function
double sigmoid(const double &x) { return 1 / (1 + exp(-x)); }

// Heaviside function
double heaviside(const double &x) { return x > 0; }

} // namespace activate
