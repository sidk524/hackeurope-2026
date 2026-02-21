import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

/**
 * createRelu — ReLU activation layer
 * Tensor slab split into left (active/green) and right (zeroed/dark) halves
 * with a green divider at x=0 marking the activation threshold
 */
export function createRelu({ units = 8 } = {}) {
  const group = new THREE.Group();
  const passColor = 0x00ff88;
  const darkColor = 0x1a1a2e;

  const w = dim(units);
  const h = 1.4;

  // Left half — active (green)
  const leftGeo = new THREE.BoxGeometry(w / 2, h, 0.26);
  const leftMat = new THREE.MeshPhongMaterial({
    color: passColor,
    emissive: new THREE.Color(passColor),
    emissiveIntensity: 0.2,
    transparent: true,
    opacity: 0.82,
    shininess: 40,
  });
  const leftMesh = new THREE.Mesh(leftGeo, leftMat);
  leftMesh.position.set(-w / 4, 0, 0);
  group.add(leftMesh);

  // Right half — zeroed (dark)
  const rightGeo = new THREE.BoxGeometry(w / 2, h, 0.26);
  const rightMat = new THREE.MeshPhongMaterial({
    color: darkColor,
    emissive: new THREE.Color(darkColor),
    emissiveIntensity: 0.02,
    transparent: true,
    opacity: 0.5,
    shininess: 10,
  });
  const rightMesh = new THREE.Mesh(rightGeo, rightMat);
  rightMesh.position.set(w / 4, 0, 0);
  group.add(rightMesh);

  // Vertical divider at x=0 — threshold line
  const divGeo = new THREE.BoxGeometry(0.06, h + 0.2, 0.35);
  const divMat = new THREE.MeshPhongMaterial({
    color: passColor,
    emissive: new THREE.Color(passColor),
    emissiveIntensity: 0.45,
    transparent: true,
    opacity: 0.9,
    shininess: 60,
  });
  const divMesh = new THREE.Mesh(divGeo, divMat);
  divMesh.position.set(0, 0, 0);
  group.add(divMesh);

  group.userData.inputAnchor  = new THREE.Vector3(-w / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( w / 2, 0, 0);

  return group;
}
