import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

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
 * createMultiHeadAttention — multi-head self-attention sublayer
 * Tensor slab visual: Q/K/V mini triples per head + output projection slab
 *
 * Anchors are strictly X-direction (y=0) so the top-to-bottom layout
 * in main.js can position this group correctly without rotation surprises.
 *
 * Internal layout (local Y = world Z after group rotation.x = π/2):
 *   y = +1.6  → head Q/K/V mini slabs (foreground when viewed from above)
 *   y =  0    → output projection slab (centre)
 *   y = −(projH/2+0.22) → dropout strip
 */
export function createMultiHeadAttention({ numHeads = 4, headSize = 32 } = {}) {
  const group = new THREE.Group();
  const color = 0xa78bfa;
  const darkColor = 0x1a1a2e;

  const headCount   = Math.min(numHeads, 4);
  const headSpacing = 2.2;
  const headRowHalfW = ((headCount - 1) * headSpacing) / 2;

  const miniW = 0.35;
  const miniH = 0.8;
  const miniD = 0.15;
  const miniGap = 0.42;
  const headsY = 1.6;

  const headCenters = [];

  for (let i = 0; i < headCount; i++) {
    const cx = i * headSpacing - headRowHalfW;
    headCenters.push(cx);

    // Three mini slabs: Q, K, V side by side
    [-miniGap, 0, miniGap].forEach(dx => {
      const geo = new THREE.BoxGeometry(miniW, miniH, miniD);
      const mesh = new THREE.Mesh(geo, slabMat(color));
      mesh.position.set(cx + dx, headsY, 0);
      group.add(mesh);
    });
  }

  // Output projection slab centred at y=0
  const nEmbd = headSize * numHeads;
  const projW = dim(Math.min(nEmbd, 512));
  const projH = projW;

  const projGeo = new THREE.BoxGeometry(projW, projH, 0.25);
  const projMesh = new THREE.Mesh(projGeo, slabMat(color));
  projMesh.position.set(0, 0, 0);
  group.add(projMesh);

  // Grid on projection slab (at y=0 centre)
  const gridPositions = [];
  const gz = 0.14;
  for (let j = 1; j < 4; j++) {
    const x = -projW / 2 + (j / 4) * projW;
    gridPositions.push(x, -projH / 2, gz,  x, projH / 2, gz);
  }
  for (let i = 1; i < 4; i++) {
    const y = -projH / 2 + (i / 4) * projH;
    gridPositions.push(-projW / 2, y, gz,  projW / 2, y, gz);
  }
  const gridGeo = new THREE.BufferGeometry();
  gridGeo.setAttribute('position', new THREE.Float32BufferAttribute(gridPositions, 3));
  group.add(new THREE.LineSegments(gridGeo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.18,
  })));

  // Lines from each head group centre down to projection slab top
  const linePositions = [];
  headCenters.forEach(cx => {
    linePositions.push(cx, headsY - miniH / 2, 0,   0, projH / 2, 0);
  });
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  group.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.35,
  })));

  // Dropout strip just below projection slab
  const dropY = -(projH / 2 + 0.22);
  const dropGeo = new THREE.BoxGeometry(projW, 0.12, 0.15);
  const dropMat = new THREE.MeshPhongMaterial({
    color: darkColor,
    emissive: new THREE.Color(darkColor),
    emissiveIntensity: 0.05,
    transparent: true,
    opacity: 0.5,
  });
  const dropMesh = new THREE.Mesh(dropGeo, dropMat);
  dropMesh.position.set(0, dropY, 0);
  group.add(dropMesh);

  // Anchors: X-direction at y=0 so main.js layout works correctly
  const halfW = Math.max(headRowHalfW + miniGap + miniW / 2, projW / 2);
  group.userData.inputAnchor  = new THREE.Vector3(-halfW, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( halfW, 0, 0);

  return group;
}
