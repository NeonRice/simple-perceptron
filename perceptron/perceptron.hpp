#include <Eigen/Dense>
#include <array>
#include <vector>

class Perceptron;

typedef std::pair<std::vector<Eigen::VectorXd>, std::vector<int>>
    training_data;

using init_fun = void (*)(Eigen::VectorXd *weights);
using active_fun = double (*)(const double &weighted_sum);
using fit_fun = bool (*)(Eigen::VectorXd *weights, Perceptron *perceptron,
                         const training_data &training_set,
                         const double &learning_rate, const ulong &epochs,
                         ulong &iterations);

class Perceptron {
public:
  Perceptron(init_fun, active_fun, fit_fun);

  std::pair<bool, ulong> train(const training_data &training_set,
                               const ulong &epochs = 250,
                               const double &learning_rate = 0.1);
  std::vector<int> predict(const std::vector<Eigen::VectorXd> &input);

  inline void set_activation_function(active_fun fun) {
    this->activation_function = fun;
  }
  inline void set_init_function(init_fun fun) {
    this->initialization_function = fun;
  }
  inline void set_fit_function(fit_fun fun) { this->fitting_function = fun; }

  inline Eigen::VectorXd get_weights() { return weights; }
  inline Eigen::VectorXd *set_weights(const Eigen::VectorXd &weights) {
    this->weights = weights;
    return &this->weights;
  }

private:
  Eigen::VectorXd weights;
  double weighted_sum = 0;

  // Functions
  init_fun initialization_function;
  active_fun activation_function;
  fit_fun fitting_function;
};
