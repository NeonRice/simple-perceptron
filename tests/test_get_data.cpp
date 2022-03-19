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
  // Label column is last
  for (auto du : data) {
    std::cout << du.first << " Label: " << du.second << std::endl;
  }
  // TODO: Convert to GTEST and finish test (static asserts)
}
