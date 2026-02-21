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

function slabMat(color) {
  return new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.08,
    transparent: true,
    opacity: 0.82,
    shininess: 40,
  });
}

function makeSlab(group, w, h, color, px, py, pz) {
  const geo = new THREE.BoxGeometry(w, h, 0.25);
  const mesh = new THREE.Mesh(geo, slabMat(color));
  mesh.position.set(px, py, pz);
  group.add(mesh);
  return mesh;
}

/**
 * createHead — single self-attention head
 * Tensor slab visual: Q/K/V slabs + attention score slab with causal mask
 */
export function createHead({ headSize = 32, blockSize = 8 } = {}) {
  const group = new THREE.Group();
  const color = 0xa78bfa;

  const qkw = dim(headSize);
  const qkh = 1.5;
  const sw  = dim(blockSize);
  const sh  = sw; // square score matrix

  // Q slab
  makeSlab(group, qkw, qkh, color, -1.5, 1.2, 0);
  addGrid(group, qkw, qkh, 4, 4, color);  // grid added at origin — shift manually below

  // K slab
  makeSlab(group, qkw, qkh, color, 0, 1.2, 0);

  // V slab
  makeSlab(group, qkw, qkh, color, 1.5, 1.2, 0);

  // Attention score slab
  makeSlab(group, sw, sh, color, 0, -0.8, 0);

  // Causal mask overlay — upper-right triangle of score slab (dark, low opacity)
  const maskGeo = new THREE.BoxGeometry(sw, sh, 0.05);
  const maskMat = new THREE.MeshPhongMaterial({
    color: 0x1a1a2e,
    emissive: new THREE.Color(0x1a1a2e),
    emissiveIntensity: 0.05,
    transparent: true,
    opacity: 0.55,
  });
  const maskMesh = new THREE.Mesh(maskGeo, maskMat);
  maskMesh.position.set(sw / 4, sh / 4, 0.14);
  group.add(maskMesh);

  // Grid lines on Q/K/V slabs and score slab
  // (approximate — place local grids relative to each slab center)
  const gridPositions = [];
  const gz = 0.14;

  // Q grid (centred at -1.5, 1.2)
  for (let j = 1; j < 4; j++) {
    const x = -1.5 - qkw / 2 + (j / 4) * qkw;
    gridPositions.push(x, 1.2 - qkh / 2, gz,  x, 1.2 + qkh / 2, gz);
  }
  for (let i = 1; i < 4; i++) {
    const y = 1.2 - qkh / 2 + (i / 4) * qkh;
    gridPositions.push(-1.5 - qkw / 2, y, gz,  -1.5 + qkw / 2, y, gz);
  }

  // K grid (centred at 0, 1.2)
  for (let j = 1; j < 4; j++) {
    const x = -qkw / 2 + (j / 4) * qkw;
    gridPositions.push(x, 1.2 - qkh / 2, gz,  x, 1.2 + qkh / 2, gz);
  }
  for (let i = 1; i < 4; i++) {
    const y = 1.2 - qkh / 2 + (i / 4) * qkh;
    gridPositions.push(-qkw / 2, y, gz,  qkw / 2, y, gz);
  }

  // V grid (centred at 1.5, 1.2)
  for (let j = 1; j < 4; j++) {
    const x = 1.5 - qkw / 2 + (j / 4) * qkw;
    gridPositions.push(x, 1.2 - qkh / 2, gz,  x, 1.2 + qkh / 2, gz);
  }
  for (let i = 1; i < 4; i++) {
    const y = 1.2 - qkh / 2 + (i / 4) * qkh;
    gridPositions.push(1.5 - qkw / 2, y, gz,  1.5 + qkw / 2, y, gz);
  }

  // Score grid (centred at 0, -0.8)
  for (let j = 1; j < 4; j++) {
    const x = -sw / 2 + (j / 4) * sw;
    gridPositions.push(x, -0.8 - sh / 2, gz,  x, -0.8 + sh / 2, gz);
  }
  for (let i = 1; i < 4; i++) {
    const y = -0.8 - sh / 2 + (i / 4) * sh;
    gridPositions.push(-sw / 2, y, gz,  sw / 2, y, gz);
  }

  const gridGeo = new THREE.BufferGeometry();
  gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(gridPositions, 3));
  group.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.18,
  })));

  // Connector lines: Q/K → score slab, score slab → V
  const linePositions = [
    // Q bottom → score top
    -1.5, 1.2 - qkh / 2, 0,   0, -0.8 + sh / 2, 0,
    // K bottom → score top
     0, 1.2 - qkh / 2, 0,     0, -0.8 + sh / 2, 0,
    // score bottom → V bottom (output)
     0, -0.8 - sh / 2, 0,     1.5, 1.2 - qkh / 2, 0,
  ];
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  group.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.4,
  })));

  group.userData.inputAnchor  = new THREE.Vector3(-2.2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( 2.2, 0, 0);

  return group;
}
