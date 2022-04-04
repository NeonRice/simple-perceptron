#include "functions/activate.hpp"
#include "functions/fit.hpp"
#include "functions/init.hpp"
#include "perceptron/perceptron.hpp"

#include "util/get_data.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  std::istream *stream;
  argc = 2;
  // argv[1] = "../versicolor0-virginica1-s0.data";
  // argv[1] = "../setosa0-other1-s2.data";
  argv[1] = "../versicolor0-virginica1-s2.data";
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
  Perceptron *perceptron =
      new Perceptron(init::initialize_weights<Perceptron::WeightVector>,
                     activate::sigmoid, fit::adaline_fit);

  std::pair<bool, ulong> results = perceptron->train(data, 0, 0.2);
  Perceptron::transformed_training_data transformed_data =
      Perceptron::transform_training_data(data);
  std::vector<double> predictions =
      perceptron->predict(transformed_data.first, false);
  bool trained = results.first;
  if (trained) {
    std::cout << "Trained model successfully!" << std::endl;
    std::cout << "Weights are given:\n"
              << perceptron->get_weights() << std::endl;
    std::cout << "Predictions:\n";
  }

  //  std::string input = "";
  //  while (std::getline(std::cin, input)) {
  //    std::vector<std::string> tokenized = util::tokenize(input);
  //    Perceptron::WeightVector input_vec =
  //        Perceptron::WeightVector(tokenized.size());
  //    std::vector<Perceptron::WeightVector> input;
  //    for (auto token : tokenized) {
  //      input_vec << stod(token);
  //    }
  //    perceptron->predict(input);
  //  }

  std::cout << "Amount of predictions " << predictions.size() << std::endl;
  for (int i = 0; i < predictions.size(); ++i) {
    std::cout << predictions[i] << " " << transformed_data.second[i]
              << std::endl;
  }

  return 0;
}
