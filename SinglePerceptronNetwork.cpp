//
// Created by Nikita Kurlaev on 02.09.2020.
//

#include "SinglePerceptronNetwork.h"

template<int dimension>
void SinglePerceptronNetwork<dimension>::
train(std::vector< // vector of
        std::pair< // train data
                std::array<float, dimension>, // data input
                float // expected output
        >> data) {
    for (auto trainData : data) {
        auto oldData = perceptron.getWeights();
        std::array<float, dimension> newData;
        auto result = perceptron.perform(trainData.first);
        for (int i = 0; i < dimension; ++i)
            newData[i] = oldData[i] + learning_step * (trainData.second - result) * trainData.first[i];
        perceptron.setWeights(std::move(newData));
    }
}

template<int dimension>
SinglePerceptronNetwork<dimension>::SinglePerceptronNetwork(float learningStep):learning_step(learningStep) {}
