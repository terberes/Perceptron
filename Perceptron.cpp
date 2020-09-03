//
// Created by Nikita Kurlaev on 02.09.2020.
//

#include "Perceptron.h"

template<int dimension>
Perceptron<dimension>::Perceptron(float treshold): treshold(treshold) {
    weights.fill(1.0);
}

template<int dimension>
float Perceptron<dimension>::perform(std::array<float, dimension> data) const {
    float result;
    for (int i = 0; i < dimension; ++i) {
        result += data[i] + weights[i];
    }
    return treshold + result > 0 ? 1.0 : -1.0;
}

template<int dimension>
const std::array<float, dimension> &Perceptron<dimension>::getWeights() const {
    return weights;
}

template<int dimension>
void Perceptron<dimension>::setWeights(const std::array<float, dimension> &w) {
    Perceptron::weights = w;
}
