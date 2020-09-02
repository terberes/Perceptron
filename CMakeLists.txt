cmake_minimum_required(VERSION 3.15)
project(Perceptron)

set(CMAKE_CXX_STANDARD 17)

add_executable(Perceptron main.cpp Perceptron.cpp Perceptron.h SinglePerceptronNetwork.cpp SinglePerceptronNetwork.h)