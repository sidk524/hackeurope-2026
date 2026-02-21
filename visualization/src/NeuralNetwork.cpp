#include "NeuralNetwork.h"
#include <random>
#include <cmath>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) {
    createNeurons(layerSizes);
    createConnections();
    randomizeWeights();
}

void NeuralNetwork::createNeurons(const std::vector<int>& layerSizes) {
    layers.resize(layerSizes.size());

    float layerSpacing = 3.0f;

    for (size_t layerIdx = 0; layerIdx < layerSizes.size(); ++layerIdx) {
        int numNeurons = layerSizes[layerIdx];
        layers[layerIdx].resize(numNeurons);

        float x = layerIdx * layerSpacing - (layerSizes.size() - 1) * layerSpacing * 0.5f;

        // Arrange neurons in a circular pattern for better 3D visualization
        for (int i = 0; i < numNeurons; ++i) {
            float angle = (2.0f * M_PI * i) / numNeurons;
            float radius = std::sqrt(numNeurons) * 0.5f;

            float y = radius * std::cos(angle);
            float z = radius * std::sin(angle);

            layers[layerIdx][i].position = glm::vec3(x, y, z);
            layers[layerIdx][i].activation = 0.5f;
        }
    }
}

void NeuralNetwork::createConnections() {
    connections.clear();

    for (size_t layerIdx = 0; layerIdx < layers.size() - 1; ++layerIdx) {
        for (size_t fromIdx = 0; fromIdx < layers[layerIdx].size(); ++fromIdx) {
            for (size_t toIdx = 0; toIdx < layers[layerIdx + 1].size(); ++toIdx) {
                Connection conn;
                conn.fromLayer = layerIdx;
                conn.fromNeuron = fromIdx;
                conn.toLayer = layerIdx + 1;
                conn.toNeuron = toIdx;
                conn.weight = 0.0f;
                connections.push_back(conn);
            }
        }
    }
}

void NeuralNetwork::randomizeWeights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (auto& conn : connections) {
        conn.weight = dis(gen);
    }
}

void NeuralNetwork::setActivations(int layer, const std::vector<float>& activations) {
    if (layer < 0 || layer >= layers.size()) return;

    for (size_t i = 0; i < std::min(activations.size(), layers[layer].size()); ++i) {
        layers[layer][i].activation = activations[i];
    }
}
