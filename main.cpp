#include "functions/activate.hpp"
#include "functions/fit.hpp"
#include "functions/init.hpp"
#include "perceptron/perceptron.hpp"

#include "util/fit_util.hpp"
#include "util/get_data.hpp"
#include "util/split_data.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  std::istream *stream;
  argc = 2;
  argv[1] = "../versicolor0-virginica1-s0.data";
  // argv[1] = "../setosa0-other1-s2.data";
  // argv[1] = "../versicolor0-virginica1-s2.data";
  if (argc == 2) {
    // File name provided
    stream = new std::ifstream(argv[1]);
    if (!stream->good()) {
      throw std::invalid_argument("Could not open file! Is the path good?");
    }
  } else {
    stream = &std::cin;
  }

  // Label column is last
  util::training_data data = util::get_training_data(stream);
  for (auto du : data) {
    std::cout << du.first << " Label: " << du.second << std::endl;
  }
  // First - testing data (20%), Second - training data (80%)
  std::pair<util::training_data, util::training_data> dataset =
      util::split(data, 0.2);

  Perceptron *perceptron =
      new Perceptron(init::initialize_weights<Perceptron::WeightVector>,
                     activate::sigmoid, fit::adaline_fit);

  std::pair<bool, ulong> results =
      perceptron->train(dataset.second, 300, 1, 0.2);

  Perceptron::transformed_training_data transformed_data =
      Perceptron::transform_training_data(dataset.second);

  std::vector<double> predictions =
      perceptron->predict(transformed_data.first, false);

  bool trained = results.first;
  if (trained) {
    std::cout << "Trained model successfully!" << std::endl;
    std::cout << "Weights are given:\n"
              << perceptron->get_weights() << std::endl;
    std::cout << "Error: "
              << util::calculate_error(predictions, transformed_data.second)
              << std::endl;
    std::cout << "Epoch count: " << results.second << std::endl;
  }

  Perceptron::transformed_training_data test_data =
      Perceptron::transform_training_data(dataset.first);

  auto test_predictions = perceptron->predict(test_data.first, false);
  std::cout << "\nTraining data prediction overview" << std::endl;
  for (size_t i = 0; i < test_predictions.size(); ++i) {
    std::cout << "Predicted: " << test_predictions[i]
              << " Actual: " << test_data.second[i] << std::endl;
  }
  std::cout << "Testing error: "
            << util::calculate_error(test_predictions, test_data.second) << std::endl;

  return 0;
}
