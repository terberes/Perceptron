#include <iostream>
#include <array>
#include "SinglePerceptronNetwork.h"

#define LEARNING_STEP 1.2
#define TEST_DIMENSIONS 2

int main() {
    auto net = SinglePerceptronNetwork<TEST_DIMENSIONS> { LEARNING_STEP };

    std::vector<std::pair<std::array<float, TEST_DIMENSIONS>, float>> train_data = {
            {{ 1.0, 0.0 }, 1.0 },
            {{ 0.0, 1.0 }, 1.0 },
            {{ 0.0, 0.0 }, 0.0 },
            {{ 1.0, 1.0 }, 1.0 }
    };

    net.train(train_data);
}