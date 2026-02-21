#version 330 core

in vec3 FragPos;
in vec3 Normal;

uniform vec3 nodeColor;
uniform float activation;

out vec4 FragColor;

void main() {
    // Simple lighting
    vec3 lightPos = vec3(10.0, 10.0, 10.0);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 norm = normalize(Normal);

    float diff = max(dot(norm, lightDir), 0.0);
    vec3 ambient = 0.3 * nodeColor;
    vec3 diffuse = diff * nodeColor;

    // Add emission based on activation
    vec3 emission = nodeColor * activation * 0.3;

    vec3 result = ambient + diffuse + emission;
    FragColor = vec4(result, 1.0);
}
