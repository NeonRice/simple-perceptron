#include "../functions/activate.hpp"
#include "../functions/fit.hpp"
#include "../functions/init.hpp"
#include "../perceptron/perceptron.hpp"

#include "../util/get_data.hpp"
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
    throw std::invalid_argument("Provide path to test.data");
  }

  util::training_data data = util::get_training_data(stream);
  delete (stream);
  Perceptron *perceptron =
      new Perceptron(init::initialize_weights<Perceptron::WeightVector>,
                     activate::sigmoid, fit::random_fit);

  std::pair<bool, ulong> results = perceptron->train(data, 0);
  std::vector<int> predictions =
      perceptron->predict(Perceptron::transform_training_data(data).first);
  bool trained = results.first;
  if (trained) {
    std::cout << "Trained model successfully!" << std::endl;
    std::cout << "Weights are given:\n"
              << perceptron->get_weights() << std::endl;
    std::cout << "Predictions:\n";
  }
  for (auto prediction : predictions) {
    std::cout << std::to_string(prediction) << std::endl;
  }

  return 0;
}
