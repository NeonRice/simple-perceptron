#include "functions/activate.hpp"
#include "functions/fit.hpp"
#include "functions/init.hpp"
#include "perceptron/dynamic_perceptron.hpp"
#include "perceptron/static_perceptron.hpp"
#include "perceptron/perceptron.hpp"

#include "util/get_data.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  argc = 2;
  argv[1] = "test.data";
  std::istream *stream;
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
  // DynamicPerceptron *perceptron = new DynamicPerceptron(
  //     init::initialize_weights<DynamicPerceptron::WeightVector>,
  //     activate::sigmoid, fit::random_fit);
  StaticPerceptron<3> *perceptron = new StaticPerceptron<3>(
      init::initialize_weights<DynamicPerceptron::WeightVector>,
      activate::sigmoid, fit::random_fit);

  //std::pair<bool, ulong> results = perceptron->train(data, 0);
  //std::vector<int> predictions =
  //    perceptron->predict(perceptron->transform_training_data(data).first);
  //bool trained = results.first;
  //if (trained) {
  //  std::cout << "Trained model successfully!" << std::endl;
  //  std::cout << "Weights are given:\n"
  //            << perceptron->get_weights() << std::endl;
  //  std::cout << "Predictions:\n";
  //}
  //for (auto prediction : predictions) {
  //  std::cout << std::to_string(prediction) << std::endl;
  //}

  DynamicPerceptron test(init::initialize_weights, activate::sigmoid,
                         fit::random_fit);

  DynamicPerceptron test(init::initialize_weights, activate::sigmoid,
                         fit::random_fit);

  //  constexpr uint k = 2;
  //  std::vector<Perceptron<k>::InputVector> input = {
  //      {-0.3, 0.6}, {0.3, -0.6}, {1.2, -1.2}, {1.2, 1.2}};
  //  std::vector<int> output = {0, 0, 1, 1};
  //
  //  Perceptron<k>::training_data training_set =
  //      Perceptron<k>::training_data(input, output);
  //
  //  Perceptron<k> *p = new Perceptron<k>(init::initialize_weights<k>,
  //                                       activate::sigmoid,
  //                                       fit::random_fit<k>);
  //  std::pair<bool, ulong> results = p->train(training_set, 0);
  //  bool trained = results.first;

  return 0;
}
