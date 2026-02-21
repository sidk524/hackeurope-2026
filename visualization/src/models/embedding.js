import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

function addGrid(group, w, h, cols, rows, color, zOffset) {
  const positions = [];
  const z = zOffset;
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
    color, transparent: true, opacity: 0.25,
  })));
}

/**
 * createEmbedding — token/position embedding lookup table W_e [vocabSize × embeddingDim]
 *
 * Rendered as a proper 3D volume (thick rectangular prism) rather than a
 * paper-thin slab, to convey that the table stores an entire vocabulary's worth
 * of vectors.  Grid lines on the front face suggest the row/column structure.
 * A second, slightly smaller face is inset to give a "layered pages" effect.
 */
export function createEmbedding({ vocabSize = 8, embeddingDim = 6 } = {}) {
  const group = new THREE.Group();
  const color = 0x00ff88;

  const w = dim(embeddingDim);
  const h = dim(vocabSize);
  const d = Math.max(0.6, h * 0.25);   // depth proportional to vocab height

  // Main body — thick prism
  const bodyGeo = new THREE.BoxGeometry(w, h, d);
  const bodyMat = new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.08,
    transparent: true,
    opacity: 0.78,
    shininess: 40,
  });
  group.add(new THREE.Mesh(bodyGeo, bodyMat));

  // Front-face grid (token rows × embedding columns)
  addGrid(group, w, h, 8, 8, color, d / 2 + 0.01);

  // Inset "page" slightly behind the front face — gives "stacked pages" look
  const pageGeo = new THREE.BoxGeometry(w * 0.92, h * 0.92, 0.04);
  const pageMat = new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.18,
    transparent: true,
    opacity: 0.55,
    shininess: 20,
  });
  const page = new THREE.Mesh(pageGeo, pageMat);
  page.position.set(0, 0, d / 2 - 0.12);
  group.add(page);

  // Grid on inset page
  addGrid(group, w * 0.92, h * 0.92, 8, 8, color, d / 2 - 0.08);

  group.userData.inputAnchor  = new THREE.Vector3(-w / 2, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( w / 2, 0, 0);

  return group;
}
