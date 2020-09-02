#pragma once

#include <vector>
#include <numeric>

template <int dimension>
class Perceptron {
    float treshold;
    std::array<float, dimension> weights = {};
public:
    const std::array<float, dimension> &getWeights() const;

    void setWeights(const std::array<float, dimension> &weights);
    explicit Perceptron(float treshold);
    float perform(std::array<float, dimension> data) const;
};

