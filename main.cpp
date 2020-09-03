#include <iostream>
#include <array>
#include <algorithm>
#include <random>
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

    net.train(train_data);

//    std::array<float, TEST_DIMENSIONS> test_data = { , 1.0 };
//
//    std::cout << "Result: " << net.perform(test_data) << "\n";
}