#pragma once
#include <vector>
#include <glm/glm.hpp>

struct Neuron {
    glm::vec3 position;
    float activation;
};

struct Connection {
    int fromLayer, fromNeuron;
    int toLayer, toNeuron;
    float weight;
};

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);

    const std::vector<std::vector<Neuron>>& getLayers() const { return layers; }
    const std::vector<Connection>& getConnections() const { return connections; }

    void setActivations(int layer, const std::vector<float>& activations);
    void randomizeWeights();

private:
    std::vector<std::vector<Neuron>> layers;
    std::vector<Connection> connections;

    void createNeurons(const std::vector<int>& layerSizes);
    void createConnections();
};
