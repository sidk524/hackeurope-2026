#!/bin/bash

echo "Downloading GLAD..."

# Create directories
mkdir -p include/glad
mkdir -p include/KHR
mkdir -p src

# Download GLAD files from a pre-generated version (OpenGL 3.3 Core)
curl -o include/glad/glad.h https://raw.githubusercontent.com/Dav1dde/glad/master/include/glad/glad.h
curl -o include/KHR/khrplatform.h https://raw.githubusercontent.com/Dav1dde/glad/master/include/KHR/khrplatform.h
curl -o src/glad.c https://raw.githubusercontent.com/Dav1dde/glad/master/src/glad.c

echo "GLAD downloaded successfully!"
echo ""
echo "Note: If the above downloads fail, you can generate GLAD manually at:"
echo "https://glad.dav1d.de/"
echo "Settings: Language=C/C++, gl=Version 3.3, Profile=Core, Generate a loader=ON"
