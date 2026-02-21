# 3D Neural Network Visualization

A real-time 3D visualization of fully connected neural networks using C++ and OpenGL.

## Features

- **3D Rendering**: Neurons displayed as spheres in 3D space
- **Interactive Camera**: Rotate and zoom to view the network from any angle
- **Layer Visualization**: Different colors for input (blue), hidden (green), and output (red) layers
- **Connection Rendering**: Weighted connections between neurons
- **Circular Layout**: Neurons arranged in circular patterns for better 3D visualization

## Prerequisites

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install build-essential cmake libglfw3-dev libglm-dev
```

### macOS
```bash
brew install cmake glfw glm
```

### Fedora/RHEL
```bash
sudo dnf install cmake glfw-devel glm-devel
```

## Building

1. **Download GLAD** (OpenGL loader):
```bash
chmod +x setup_glad.sh
./setup_glad.sh
```

2. **Build the project**:
```bash
mkdir build
cd build
cmake ..
make
```

3. **Run**:
```bash
./NeuralNetVisualization
```

## Controls

- **Left Mouse Drag**: Rotate camera around the network
- **Mouse Wheel**: Zoom in/out
- **ESC**: Exit application

## Network Architecture

Default network: `[4, 6, 6, 3]`
- 4 input neurons
- 2 hidden layers with 6 neurons each
- 3 output neurons

You can modify the architecture in `src/main.cpp` by changing the layer sizes:
```cpp
NeuralNetwork network({4, 6, 6, 3}); // Change these numbers
```

## Project Structure

```
visualization/
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
│   ├── NeuralNetwork.h    # Network structure
│   ├── Renderer.h         # OpenGL rendering
│   ├── Camera.h           # 3D camera
│   └── Shader.h           # Shader management
├── src/                   # Source files
│   ├── main.cpp           # Entry point
│   ├── NeuralNetwork.cpp
│   ├── Renderer.cpp
│   ├── Camera.cpp
│   ├── Shader.cpp
│   └── glad.c             # OpenGL loader
└── shaders/               # GLSL shaders
    ├── node.vert
    ├── node.frag
    ├── connection.vert
    └── connection.frag
```

## Customization

### Change Colors
Edit the colors in `src/Renderer.cpp` in the `renderNeurons()` function.

### Add Animation
You can add forward pass animation by:
1. Updating neuron activations over time
2. Animating connection weights
3. Adding particle effects for signal propagation

### Network Layout
Modify `createNeurons()` in `src/NeuralNetwork.cpp` to change how neurons are positioned.

## Troubleshooting

**GLAD download fails**: Manually generate GLAD at https://glad.dav1d.de/
- Language: C/C++
- gl: Version 3.3
- Profile: Core
- Generate a loader: ON

**GLFW/GLM not found**: Install dependencies using your package manager (see Prerequisites)

**Black screen**: Check that your graphics drivers support OpenGL 3.3+

## Future Enhancements

- [ ] Animated forward/backward pass
- [ ] Load network weights from file
- [ ] Real-time training visualization
- [ ] Gradient flow visualization
- [ ] Support for convolutional layers
- [ ] Interactive neuron selection
- [ ] Performance metrics overlay
