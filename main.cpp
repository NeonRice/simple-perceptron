#include "functions/activate.hpp"
#include "functions/fit.hpp"
#include "functions/init.hpp"
#include "perceptron/perceptron.hpp"
#include <iostream>

int main(int argc, char **argv) {
  constexpr uint k = 2;
  std::vector<Perceptron<k>::InputVector> input = {
      {-0.3, 0.6}, {0.3, -0.6}, {1.2, -1.2}, {1.2, 1.2}};
  std::vector<int> output = {0, 0, 1, 1};

  Perceptron<k>::training_data training_set =
      Perceptron<k>::training_data(input, output);

  Perceptron<k> *p = new Perceptron<k>(init::initialize_weights<k>,
                                       activate::sigmoid, fit::random_fit<k>);
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
