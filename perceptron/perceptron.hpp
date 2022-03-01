#include <array>
#include <vector>

class Perceptron;

using init_fun = void (*)(std::array<double, 3> *weights);
using active_fun = double (*)(const double &weighted_sum);
using fit_fun = bool (*)(std::array<double, 3> *weights, Perceptron *perceptron,
                         const std::vector<std::array<double, 2>> &input,
                         const std::vector<int> &output,
                         const double &learning_rate, const ulong &epochs,
                         ulong &iterations);

class Perceptron {
public:
  Perceptron(init_fun, active_fun, fit_fun);

  std::pair<bool, ulong> train(const std::vector<std::array<double, 2>> &input,
                               const std::vector<int> &output,
                               const ulong &epochs = 250, const double &learning_rate = 0.1);
  std::vector<int> predict(const std::vector<std::array<double, 2>> &input);

  inline void set_activation_function(active_fun fun) {
    this->activation_function = fun;
  }
  inline void set_init_function(init_fun fun) {
    this->initialization_function = fun;
  }
  inline void set_fit_function(fit_fun fun) { this->fitting_function = fun; }

  std::array<double, 3> get_weights() { return weights; }
  std::array<double, 3> *set_weights(std::array<double, 3> weights) {
    this->weights = weights;
    return &this->weights;
  }

private:
  std::array<double, 3> weights;
  double weighted_sum = 0;

  // Functions
  init_fun initialization_function;
  active_fun activation_function;
  fit_fun fitting_function;
};
