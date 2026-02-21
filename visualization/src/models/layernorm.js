import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

function addGrid(group, w, h, cols, rows, color) {
  const positions = [];
  const z = 0.14;
  for (let i = 1; i < rows; i++) {
    const y = -h / 2 + (i / rows) * h;
    positions.push(-w / 2, y, z,  w / 2, y, z);
  }
  for (let j = 1; j < cols; j++) {
    const x = -w / 2 + (j / cols) * w;
    positions.push(x, -h / 2, z,  x, h / 2, z);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  group.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.18,
  })));
}

/**
 * createLayerNorm — layer normalisation
 * Tensor slab visual: thin horizontal strip (1D per-feature scale/shift)
 */
export function createLayerNorm({ normalizedDim = 8 } = {}) {
  const group = new THREE.Group();
  const color = 0xffa500;

  const w = dim(normalizedDim);
  const h = 0.35;

  const geo = new THREE.BoxGeometry(w, h, 0.25);
  const mat = new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.25,
    transparent: true,
    opacity: 0.82,
    shininess: 40,
  });
  const mesh = new THREE.Mesh(geo, mat);
  group.add(mesh);

  addGrid(group, w, h, 6, 1, color);

  group.userData.inputAnchor  = new THREE.Vector3(-w / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( w / 2, 0, 0);

  return group;
}
