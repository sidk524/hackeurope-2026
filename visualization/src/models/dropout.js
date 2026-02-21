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

/**
 * createDropout — dropout regularisation layer
 * Tensor slab with dark overlay cells for masked units
 */
export function createDropout({ units = 8, dropRate = 0.2 } = {}) {
  const group = new THREE.Group();
  const color = 0x4ecdc4;
  const maskColor = 0x1a1a2e;

  const w = dim(units);
  const h = 1.4;

  const geo = new THREE.BoxGeometry(w, h, 0.25);
  const mesh = new THREE.Mesh(geo, slabMat(color));
  group.add(mesh);

  const cols = 6;
  const rows = 4;
  addGrid(group, w, h, cols, rows, color);

  const cellW = w / cols;
  const cellH = h / rows;
  const maskCount = Math.floor(cols * rows * dropRate);

  // Place masked cell overlays (deterministic, top-left first)
  for (let idx = 0; idx < maskCount; idx++) {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const cx = -w / 2 + (col + 0.5) * cellW;
    const cy = -h / 2 + (row + 0.5) * cellH;
    const cellGeo = new THREE.BoxGeometry(cellW * 0.9, cellH * 0.9, 0.05);
    const cellMat = new THREE.MeshPhongMaterial({
      color: maskColor,
      emissive: new THREE.Color(maskColor),
      emissiveIntensity: 0.05,
      transparent: true,
      opacity: 0.85,
    });
    const cellMesh = new THREE.Mesh(cellGeo, cellMat);
    cellMesh.position.set(cx, cy, 0.15);
    group.add(cellMesh);
  }

  group.userData.inputAnchor  = new THREE.Vector3(-w / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( w / 2, 0, 0);

  return group;
}
