cmake_minimum_required(VERSION 3.15)
project(NeuralNetVisualization VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenGL
find_package(OpenGL REQUIRED)

# GLFW
find_package(glfw3 3.3 QUIET)
if(NOT glfw3_FOUND)
    message(STATUS "GLFW not found, will need to install it")
endif()

# GLM
find_package(glm QUIET)
if(NOT glm_FOUND)
    message(STATUS "GLM not found, will need to install it")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/glad.c
    src/NeuralNetwork.cpp
    src/Renderer.cpp
    src/Camera.cpp
    src/Shader.cpp
)

set(HEADERS
    include/NeuralNetwork.h
    include/Renderer.h
    include/Camera.h
    include/Shader.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OPENGL_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    OpenGL::GL
    glfw
    ${CMAKE_DL_LIBS}
)

# Copy shaders to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
