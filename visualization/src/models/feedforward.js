import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

function addGrid(group, w, h, cols, rows, color, ox, oy) {
  const positions = [];
  const z = 0.14;
  for (let i = 1; i < rows; i++) {
    const y = oy - h / 2 + (i / rows) * h;
    positions.push(ox - w / 2, y, z,  ox + w / 2, y, z);
  }
  for (let j = 1; j < cols; j++) {
    const x = ox - w / 2 + (j / cols) * w;
    positions.push(x, oy - h / 2, z,  x, oy + h / 2, z);
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
 * createFeedForward — feed-forward (MLP) sublayer
 * Tensor slab visual: expand slab | ReLU divider | contract slab | dropout strip
 */
export function createFeedForward({ nEmbd = 8, expansion = 4 } = {}) {
  const group = new THREE.Group();
  const color = 0xf472b6;
  const reluColor = 0x00ff88;
  const darkColor = 0x1a1a2e;

  const expandW = dim(nEmbd * expansion);
  const expandH = dim(nEmbd);
  const contractW = dim(nEmbd);
  const contractH = dim(nEmbd * expansion);

  const expandX   = -2.2;
  const contractX =  2.2;

  // Expand slab [nEmbd × nEmbd*expansion]
  const expandGeo = new THREE.BoxGeometry(expandW, expandH, 0.25);
  const expandMesh = new THREE.Mesh(expandGeo, slabMat(color));
  expandMesh.position.set(expandX, 0, 0);
  group.add(expandMesh);
  addGrid(group, expandW, expandH, 4, 4, color, expandX, 0);

  // Contract slab [nEmbd*expansion × nEmbd]
  const contractGeo = new THREE.BoxGeometry(contractW, contractH, 0.25);
  const contractMesh = new THREE.Mesh(contractGeo, slabMat(color));
  contractMesh.position.set(contractX, 0, 0);
  group.add(contractMesh);
  addGrid(group, contractW, contractH, 4, 4, color, contractX, 0);

  // ReLU divider at x=0
  const divH = Math.max(expandH, contractH) + 0.3;
  const divGeo = new THREE.BoxGeometry(0.08, divH, 0.3);
  const divMat = new THREE.MeshPhongMaterial({
    color: reluColor,
    emissive: new THREE.Color(reluColor),
    emissiveIntensity: 0.45,
    transparent: true,
    opacity: 0.9,
    shininess: 60,
  });
  const divMesh = new THREE.Mesh(divGeo, divMat);
  divMesh.position.set(0, 0, 0);
  group.add(divMesh);

  // Connector line: expand right edge → contract left edge
  const linePositions = [
    expandX + expandW / 2, 0, 0,   contractX - contractW / 2, 0, 0,
  ];
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
  group.add(new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
    color, transparent: true, opacity: 0.4,
  })));

  // Dropout strip to the right of contract slab
  const dropX = contractX + contractW / 2 + 0.18;
  const dropGeo = new THREE.BoxGeometry(0.12, contractH, 0.15);
  const dropMat = new THREE.MeshPhongMaterial({
    color: darkColor,
    emissive: new THREE.Color(darkColor),
    emissiveIntensity: 0.05,
    transparent: true,
    opacity: 0.5,
  });
  const dropMesh = new THREE.Mesh(dropGeo, dropMat);
  dropMesh.position.set(dropX, 0, 0);
  group.add(dropMesh);

  group.userData.inputAnchor  = new THREE.Vector3(expandX - expandW / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3(dropX + 0.06, 0, 0);

  return group;
}
