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
 * createLinear — fully-connected (linear/dense) layer
 * Tensor slab visual: W [inFeatures × outFeatures]
 */
export function createLinear({ inFeatures = 6, outFeatures = 8, showBias = false } = {}) {
  const group = new THREE.Group();
  const color = 0x4ecdc4;

  const w = dim(outFeatures);
  const h = dim(inFeatures);

  const geo = new THREE.BoxGeometry(w, h, 0.25);
  const mesh = new THREE.Mesh(geo, slabMat(color));
  group.add(mesh);

  addGrid(group, w, h, 5, 5, color);

  if (showBias) {
    const bh = 0.18;
    const biasGeo = new THREE.BoxGeometry(w, bh, 0.15);
    const biasMat = new THREE.MeshPhongMaterial({
      color,
      emissive: new THREE.Color(color),
      emissiveIntensity: 0.22,
      transparent: true,
      opacity: 0.75,
      shininess: 40,
    });
    const biasMesh = new THREE.Mesh(biasGeo, biasMat);
    biasMesh.position.set(0, -(h / 2 + bh / 2 + 0.05), 0);
    group.add(biasMesh);
  }

  group.userData.inputAnchor  = new THREE.Vector3(-w / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( w / 2, 0, 0);

  return group;
}
