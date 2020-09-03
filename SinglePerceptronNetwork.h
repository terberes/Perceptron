#pragma once

#include "Perceptron.h"
#include <array>

#define DEFAULT_PERCEPTRON_TRESHOLD 1

template <int dimension>
class SinglePerceptronNetwork {
    float learning_step{};
    Perceptron<dimension> perceptron = Perceptron<2>{DEFAULT_PERCEPTRON_TRESHOLD};

public:
    SinglePerceptronNetwork() = default;

    explicit SinglePerceptronNetwork(float learningStep);

    void train(std::vector<std::pair<std::array<float, dimension>, float>> data);
    float perform(std::array<float, dimension> data);
};

template class SinglePerceptronNetwork<2>;
