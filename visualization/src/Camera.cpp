#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

Camera::Camera(float fov, float aspectRatio, float nearPlane, float farPlane)
    : fov(fov)
    , aspectRatio(aspectRatio)
    , nearPlane(nearPlane)
    , farPlane(farPlane)
    , radius(15.0f)
    , theta(0.0f)
    , phi(0.0f)
    , target(0.0f, 0.0f, 0.0f)
    , up(0.0f, 1.0f, 0.0f)
    , sensitivity(0.005f)
    , zoomSensitivity(1.0f)
{
}

void Camera::processMouseMovement(float xoffset, float yoffset) {
    theta += xoffset * sensitivity;
    phi += yoffset * sensitivity;

    // Clamp phi to prevent camera flipping
    phi = std::clamp(phi, -1.5f, 1.5f);
}

void Camera::processMouseScroll(float yoffset) {
    radius -= yoffset * zoomSensitivity;
    radius = std::clamp(radius, 2.0f, 50.0f);
}

glm::mat4 Camera::getViewMatrix() const {
    // Spherical to Cartesian coordinates
    float x = radius * std::cos(phi) * std::sin(theta);
    float y = radius * std::sin(phi);
    float z = radius * std::cos(phi) * std::cos(theta);

    glm::vec3 position = target + glm::vec3(x, y, z);

    return glm::lookAt(position, target, up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(fov), aspectRatio, nearPlane, farPlane);
}

void Camera::setAspectRatio(float newAspectRatio) {
    aspectRatio = newAspectRatio;
}
