#include <iostream>
#include <array>
#include <algorithm>
#include <random>
#include <iterator>
#include "SinglePerceptronNetwork.h"
#include "Perceptron.h"

#define LEARNING_STEP 0.05
#define TEST_DIMENSIONS 2
#define LEARINIG_DATA_COUNT 10

int main() {
    auto net = SinglePerceptronNetwork<TEST_DIMENSIONS>{ LEARNING_STEP };

    std::cout << "Generating training data" << "\n";

    std::vector<std::pair<std::array<float, TEST_DIMENSIONS>, float>> train_data{};

    std::random_device device{};
    std::default_random_engine random{device()};

    std::uniform_real_distribution<float> group_first_x(-10.0, 0.0);
    std::uniform_real_distribution<float> group_first_y(0.0, 10.0);

    for (int i = 0; i < LEARINIG_DATA_COUNT / 2; ++i) {
        auto x = group_first_x(random);
        auto y = group_first_y(random);
        train_data.push_back({{x, y}, 1.0});
    }

    std::uniform_real_distribution<float> group_second_x(0.0, 10.0);
    std::uniform_real_distribution<float> group_second_y(-10.0, 0.0);

    for (int i = 0; i < LEARINIG_DATA_COUNT / 2; ++i) {
        auto x = group_second_x(random);
        auto y = group_second_y(random);
        train_data.push_back({{x, y}, -1.0});
    }

    std::shuffle(train_data.begin(), train_data.end(), random);

    std::cout << "Starting model training process" << "\n";

    for (auto train_data_pair: train_data) {
        std::cout << "{ ";
        std::copy(train_data_pair.first.begin(),
                  train_data_pair.first.end(),
                  std::ostream_iterator<int>(std::cout, " "));
        std::cout << "}: " << train_data_pair.second << "\n";
    }

    net.train(train_data);
}