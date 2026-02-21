#pragma once
#include <glm/glm.hpp>

class Camera {
public:
    Camera(float fov, float aspectRatio, float nearPlane, float farPlane);

    void processMouseMovement(float xoffset, float yoffset);
    void processMouseScroll(float yoffset);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;

    void setAspectRatio(float aspectRatio);

private:
    float radius;
    float theta;
    float phi;

    glm::vec3 target;
    glm::vec3 up;

    float fov;
    float aspectRatio;
    float nearPlane;
    float farPlane;

    float sensitivity;
    float zoomSensitivity;
};
