#include "functions/activate.hpp"
#include "functions/fit.hpp"
#include "functions/init.hpp"
#include "perceptron/perceptron.hpp"

#include "util/get_data.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
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
