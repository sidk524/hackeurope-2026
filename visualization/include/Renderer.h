#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include "NeuralNetwork.h"
#include "Camera.h"
#include "Shader.h"
#include <memory>

class Renderer {
public:
    Renderer();
    ~Renderer();

    void initialize();
    void render(const NeuralNetwork& network, const Camera& camera);
    void cleanup();

private:
    std::unique_ptr<Shader> nodeShader;
    std::unique_ptr<Shader> connectionShader;

    GLuint sphereVAO, sphereVBO, sphereEBO;
    GLuint lineVAO, lineVBO;

    int sphereIndexCount;

    void createSphere(float radius, int segments);
    void renderNeurons(const NeuralNetwork& network, const glm::mat4& view, const glm::mat4& projection);
    void renderConnections(const NeuralNetwork& network, const glm::mat4& view, const glm::mat4& projection);
};
