#include "../util/split_data.hpp"
#include <fstream>
#include <iostream>

void print_vector(const std::vector<int>& vec) {
  for (auto element : vec) {
    std::cout << element << " ";
  }
  std::cout << std::endl;
}


int main(int argc, char **argv) {
  std::vector<int> test_vector = {5, 3, 1, 4};
  std::vector<int> expected_part1 = {5}, expected_part2 = {3, 1, 4};
  auto test_result = util::split(test_vector, 0.2);

  std::cout << "Expected part1: ";
  print_vector(expected_part1);
  std::cout << "Result: ";
  print_vector(test_result.first);

  std::cout << "Expected part2: ";
  print_vector(expected_part2);
  std::cout << "Result: ";
  print_vector(test_result.second);

  assert(test_result.first == expected_part1);
  assert(test_result.second == expected_part2);
}
