
#include "SinglePerceptronNetwork.h"
#include <iostream>
#include <iterator>

template<int dimension>
void SinglePerceptronNetwork<dimension>::
train(std::vector< // vector of
        std::pair< // train data
                std::array<float, dimension>, // data input
                float // expected output
        >> data) {
    bool error = false;
    int step = 0;
    do {
        std::cout << "\nTraining step " << ++step << "\n";
        bool internalTrainingErr = false;
        for (auto trainData : data) {
            std::cout << "Training on dataset: { ";
            std::copy(trainData.first.begin(),
                      trainData.first.end(),
                      std::ostream_iterator<int>(std::cout, " "));
            std::cout << "}, expected result " << trainData.second << "\n";
            auto oldWeights = perceptron.getWeights();
            std::array<float, dimension> newWeights;
            auto result = perceptron.perform(trainData.first);
            std::cout << "Actual result: " << result << "\n";
            for (int i = 0; i < dimension; ++i) {
                newWeights[i] = oldWeights[i] + learning_step * (trainData.second - result) * trainData.first[i];
                std::cout << "Using formula: " << newWeights[i] << " = " << oldWeights[i] << " + " <<
                          learning_step << " * (" << trainData.second << " - " << result << ") * " << trainData.first[i]
                          << "\n";
                std::cout << i << " :: Changing old weight value : " << oldWeights[i] << " to " << newWeights[i]
                          << "\n";
            }
            perceptron.setWeights(std::move(newWeights));
            if (result != trainData.second)
                internalTrainingErr = true;
        }
        error = internalTrainingErr;
    } while (error);
    std::cout << "Model successfully trained!" << "\n";
}

template<int dimension>
SinglePerceptronNetwork<dimension>::
SinglePerceptronNetwork(float learningStep)
        : learning_step(learningStep) { }

template<int dimension>
float SinglePerceptronNetwork<dimension>::perform(std::array<float, dimension> data) {
    return perceptron.perform(data);
}
