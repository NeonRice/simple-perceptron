#pragma once

#include "Eigen/Dense"
#include <iostream>
#include <vector>

namespace util {
typedef std::vector<std::pair<Eigen::VectorXd, int>> training_data;

std::vector<std::string> tokenize(const std::string &str,
                                  const char &delim = ',') {
  size_t start;
  size_t end = 0;
  std::vector<std::string> out;

  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    out.push_back(str.substr(start, end - start));
  }

  return out;
}

bool is_number(const std::string &s) {
  char *end = nullptr;
  double val = strtod(s.c_str(), &end);
  return end != s.c_str() && *end == '\0' && val != HUGE_VAL;
}

training_data get_training_data(std::istream *stream, uint label_column = 0) {
  training_data data;
  int previous_length = -1;
  uint line_no = 0;

  for (std::string line; std::getline(*(stream), line);) {
    ++line_no;
    std::vector<std::string> tokens = tokenize(line);
    if (tokens.size() == 0) {
      continue;
    }

    Eigen::VectorXd weights(tokens.size() - 1);
    int label;

    if (label_column == 0 && is_number(tokens.back())) {
      label = stod(tokens.back());
      tokens.pop_back();
    } else if (label_column != 0 && is_number(tokens[label_column - 1])) {
      label = stod(tokens[label_column - 1]);
      tokens.erase(tokens.begin() + (label_column - 1));
    }

    if (previous_length != -1 && previous_length != tokens.size()) {
      throw std::invalid_argument(
          "Data malformed at line no. " + std::to_string(line_no) +
          ". Previous amount of columns: " + std::to_string(previous_length) +
          ". Current line columns: " + std::to_string(tokens.size()));
    }


    for (uint i = 0; i < tokens.size(); ++i) {
      if (!is_number(tokens[i])) {
        throw std::invalid_argument("The data values must be scalar");
      }
      weights[i] = stod(tokens[i]);
    }
    data.push_back({weights, label});
    previous_length = tokens.size();
  }
  return data;
}
} // namespace util
