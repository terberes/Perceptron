#pragma once

#include "Perceptron.h"
#include <array>

template <int dimension>
class SinglePerceptronNetwork {
    float learning_step;
    Perceptron<dimension> perceptron = {};

public:
    SinglePerceptronNetwork() = default;

    explicit SinglePerceptronNetwork(float learningStep);

    void train(std::vector<std::pair<std::array<float, dimension>, float>> data);
};
