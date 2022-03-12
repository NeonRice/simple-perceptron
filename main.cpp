#include "perceptron/perceptron.hpp"
#include <array>
#include <climits>
#include <cmath>
#include <iostream>
#include <random>

double get_random(const double &from, const double &to) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(from, to);
  return dist(gen);
}

void initialize_weights(Eigen::VectorXd *weights) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  for (double &weight : *weights) {
    weight = dist(gen);
  }
}

// Fast sigmoid function
double fast_sigmoid(const double &x) { return x / (1 + fabs(x)); }

// Classical sigmoid function
double sigmoid(const double &x) { return 1 / (1 + exp(-x)); }

// Heaviside function
double heaviside(const double &x) { return x > 0; }

bool random_fit(Eigen::VectorXd *weights, Perceptron *perceptron,
                const training_data &training_set, const double &learning_rate,
                const ulong &epochs, ulong &iterations) {
  std::mt19937 gen(time(NULL));
  std::uniform_real_distribution<double> dist(-3, 3);
  while (perceptron->predict(training_set.first) != training_set.second) {
    if (++iterations != ULONG_MAX && epochs == iterations && epochs != 0) {
      // Training failed, not enough epochs?
      // If epochs = 0, never stop until weights found
      return false;
    }
    for (double &weight : *weights) {
      weight = dist(gen);
    }
  }
  return true;
}

using Eigen::Matrix;

int main(int argc, char **argv) {
  std::vector<Matrix<double, 1, 2>> input {
      {-0.3, 0.6}, {0.3, -0.6}, {1.2, -1.2}, {1.2, 1.2}};
  std::vector<int> output = {0, 0, 1, 1};

  training_data training_set = training_data(input, output);

  Perceptron *p = new Perceptron(initialize_weights, sigmoid, random_fit);
  std::pair<bool, ulong> results = p->train(training_set, 0);
  bool trained = results.first;
  if (trained) {
    std::cout << "Successfully trained perceptron on the given dataset:\n";
  } else {
    std::cout << "Failed training perceptron on the given dataset:\n";
  }

  for (int i = 0; i < input.size(); ++i) {
    std::cout << input[i][0] << " " << input[i][1] << " " << output[i]
              << std::endl;
  }
  std::cout << "Took " << results.second << " epochs to train." << std::endl;

  if (!trained) {
    return -1;
  }

  auto trained_weights = p->get_weights();
  std::cout << "Weights: " << trained_weights[0] << " " << trained_weights[1]
            << " " << trained_weights[2] << std::endl;

  auto result = p->predict(input);
  for (auto out : result) {
    std::cout << out << std::endl;
  }
  return 0;
}
