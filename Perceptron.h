#pragma once

#include <vector>
#include <array>

template <int dimension>
class Perceptron {
    float treshold;
    std::array<float, dimension> weights = {};
public:
    [[nodiscard]] const std::array<float, dimension> &getWeights() const;

    void setWeights(const std::array<float, dimension> &weights);
    explicit Perceptron(float treshold);
    [[nodiscard]] float perform(std::array<float, dimension> data) const;
};

template class Perceptron<2>;

