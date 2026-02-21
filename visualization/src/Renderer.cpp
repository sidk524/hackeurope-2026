#include "Renderer.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <vector>

Renderer::Renderer()
    : sphereVAO(0), sphereVBO(0), sphereEBO(0)
    , lineVAO(0), lineVBO(0)
    , sphereIndexCount(0)
{
}

Renderer::~Renderer() {
    cleanup();
}

void Renderer::initialize() {
    // Create shaders
    nodeShader = std::make_unique<Shader>("shaders/node.vert", "shaders/node.frag");
    connectionShader = std::make_unique<Shader>("shaders/connection.vert", "shaders/connection.frag");

    // Create sphere geometry for neurons
    createSphere(0.3f, 20);

    // Create line VAO for connections
    glGenVertexArrays(1, &lineVAO);
    glGenBuffers(1, &lineVBO);

    glBindVertexArray(lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
    glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
}

void Renderer::createSphere(float radius, int segments) {
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // Generate sphere vertices
    for (int lat = 0; lat <= segments; ++lat) {
        float theta = lat * M_PI / segments;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        for (int lon = 0; lon <= segments; ++lon) {
            float phi = lon * 2.0f * M_PI / segments;
            float sinPhi = std::sin(phi);
            float cosPhi = std::cos(phi);

            float x = cosPhi * sinTheta;
            float y = cosTheta;
            float z = sinPhi * sinTheta;

            vertices.push_back(x * radius);
            vertices.push_back(y * radius);
            vertices.push_back(z * radius);

            // Normals (same as position for unit sphere)
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }

    // Generate indices
    for (int lat = 0; lat < segments; ++lat) {
        for (int lon = 0; lon < segments; ++lon) {
            int first = lat * (segments + 1) + lon;
            int second = first + segments + 1;

            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
        }
    }

    sphereIndexCount = indices.size();

    // Create buffers
    glGenVertexArrays(1, &sphereVAO);
    glGenBuffers(1, &sphereVBO);
    glGenBuffers(1, &sphereEBO);

    glBindVertexArray(sphereVAO);

    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void Renderer::render(const NeuralNetwork& network, const Camera& camera) {
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = camera.getProjectionMatrix();

    renderConnections(network, view, projection);
    renderNeurons(network, view, projection);
}

void Renderer::renderNeurons(const NeuralNetwork& network, const glm::mat4& view, const glm::mat4& projection) {
    nodeShader->use();
    nodeShader->setMat4("view", view);
    nodeShader->setMat4("projection", projection);

    glBindVertexArray(sphereVAO);

    const auto& layers = network.getLayers();
    for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx) {
        for (size_t neuronIdx = 0; neuronIdx < layers[layerIdx].size(); ++neuronIdx) {
            const Neuron& neuron = layers[layerIdx][neuronIdx];

            glm::mat4 model = glm::translate(glm::mat4(1.0f), neuron.position);
            nodeShader->setMat4("model", model);

            // Color based on layer (input=blue, hidden=green, output=red)
            glm::vec3 color;
            if (layerIdx == 0) {
                color = glm::vec3(0.3f, 0.5f, 1.0f); // Blue for input
            } else if (layerIdx == layers.size() - 1) {
                color = glm::vec3(1.0f, 0.3f, 0.3f); // Red for output
            } else {
                color = glm::vec3(0.3f, 1.0f, 0.5f); // Green for hidden
            }

            nodeShader->setVec3("nodeColor", color);
            nodeShader->setFloat("activation", neuron.activation);

            glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
        }
    }

    glBindVertexArray(0);
}

void Renderer::renderConnections(const NeuralNetwork& network, const glm::mat4& view, const glm::mat4& projection) {
    connectionShader->use();
    connectionShader->setMat4("view", view);
    connectionShader->setMat4("projection", projection);
    connectionShader->setMat4("model", glm::mat4(1.0f));

    glBindVertexArray(lineVAO);

    const auto& layers = network.getLayers();
    const auto& connections = network.getConnections();

    for (const auto& conn : connections) {
        const glm::vec3& from = layers[conn.fromLayer][conn.fromNeuron].position;
        const glm::vec3& to = layers[conn.toLayer][conn.toNeuron].position;

        float lineData[6] = {
            from.x, from.y, from.z,
            to.x, to.y, to.z
        };

        glBindBuffer(GL_ARRAY_BUFFER, lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lineData), lineData);

        // Color based on weight (positive=white, negative=dark)
        float weight = conn.weight;
        float alpha = std::min(std::abs(weight), 1.0f) * 0.3f;
        glm::vec4 color = weight > 0
            ? glm::vec4(0.8f, 0.8f, 0.8f, alpha)
            : glm::vec4(0.4f, 0.4f, 0.5f, alpha);

        connectionShader->setVec4("lineColor", color);

        glDrawArrays(GL_LINES, 0, 2);
    }

    glBindVertexArray(0);
}

void Renderer::cleanup() {
    if (sphereVAO) glDeleteVertexArrays(1, &sphereVAO);
    if (sphereVBO) glDeleteBuffers(1, &sphereVBO);
    if (sphereEBO) glDeleteBuffers(1, &sphereEBO);
    if (lineVAO) glDeleteVertexArrays(1, &lineVAO);
    if (lineVBO) glDeleteBuffers(1, &lineVBO);
}
