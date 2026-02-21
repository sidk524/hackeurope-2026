import * as THREE from 'three';

function dim(n) { return Math.max(0.7, Math.min(3.2, Math.log2(n) * 0.48)); }

function mat(color, opacity = 0.92, glow = 0.07) {
  return new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: glow,
    transparent: true,
    opacity,
    shininess: 40,
  });
}

function addRod(group, a, b, color = 0x4ecdc4, radius = 0.045, opacity = 0.5) {
  const from = new THREE.Vector3(a[0], a[1], a[2]);
  const to = new THREE.Vector3(b[0], b[1], b[2]);
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  if (len < 1e-4) return;
  const geo = new THREE.CylinderGeometry(radius, radius, len, 8);
  const mat = new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.1,
    transparent: true,
    opacity,
  });
  const rod = new THREE.Mesh(geo, mat);
  rod.position.copy(from.clone().add(to).multiplyScalar(0.5));
  rod.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  group.add(rod);
}

function addVolumeStack(group, x, y, z, w, h, d, color, layers = 2) {
  for (let i = 0; i < layers; i++) {
    const s = 1 - i * 0.06;
    const geo = new THREE.BoxGeometry(w * s, h * s, d * s);
    const mesh = new THREE.Mesh(geo, mat(color, 0.9 - i * 0.1, 0.08 + i * 0.04));
    mesh.position.set(x, y, z - i * 0.1);
    group.add(mesh);
  }
}

export function createBlock({ nEmbd = 8, numHeads = 4 } = {}) {
  const group = new THREE.Group();

  const TOP_Y = 1.9;
  const MID_Y = 1.0;
  const LOW_Y = 0.2;
  const SKIP_Y = -1.3;
  const FRONT_Z = 0.65;
  const MID_Z = 0.0;
  const BACK_Z = -0.65;

  const LN1_X = -5.9;
  const ln1W = dim(nEmbd);
  addVolumeStack(group, LN1_X, TOP_Y, MID_Z, ln1W, 0.4, 0.7, 0xffa500, 1);

  const MHA_X = -3.45;
  const headCount = Math.min(numHeads, 4);
  const miniW = 0.3;
  const miniH = 0.58;
  const miniD = 0.32;
  const miniGap = 0.35;
  const headSpacingBlock = 1.06;
  const mhaHalfW = ((headCount - 1) * headSpacingBlock) / 2;
  const headYs = [TOP_Y + 0.2, TOP_Y - 0.1, TOP_Y - 0.4];
  for (let i = 0; i < headCount; i++) {
    const hx = MHA_X + i * headSpacingBlock - mhaHalfW;
    [-miniGap, 0, miniGap].forEach((dx, j) => {
      const geo = new THREE.BoxGeometry(miniW, miniH, miniD);
      const mesh = new THREE.Mesh(geo, mat(0xa78bfa, 0.9, 0.1));
      mesh.position.set(hx + dx, headYs[j], FRONT_Z - j * 0.36);
      group.add(mesh);
    });
  }

  const scoreW = dim(8.5);
  const scoreH = scoreW * 0.88;
  addVolumeStack(group, MHA_X, MID_Y, MID_Z, scoreW, scoreH, 0.55, 0xa78bfa, 2);

  const PROJ_X = -1.25;
  const projW = dim(nEmbd);
  const projH = dim(nEmbd);
  addVolumeStack(group, PROJ_X, MID_Y, MID_Z, projW, projH, 0.8, 0x4ecdc4, 2);

  const LN2_X = 0.5;
  addVolumeStack(group, LN2_X, MID_Y, MID_Z, ln1W, 0.4, 0.7, 0xffa500, 1);

  const FFN_EXP_X = 2.95;
  const ffnExpW = dim(nEmbd * 4);
  const ffnExpH = dim(nEmbd) * 0.9;
  addVolumeStack(group, FFN_EXP_X, LOW_Y, BACK_Z, ffnExpW, ffnExpH, 0.95, 0xf472b6, 2);

  const RELU_X = 4.65;
  const reluH = ffnExpH + 0.3;
  const reluGeo = new THREE.BoxGeometry(0.12, reluH, 0.95);
  const reluMat = mat(0x00ff88, 0.94, 0.45);
  const reluMesh = new THREE.Mesh(reluGeo, reluMat);
  reluMesh.position.set(RELU_X, LOW_Y, BACK_Z);
  group.add(reluMesh);

  const FFN_CON_X = 6.25;
  const ffnConW = dim(nEmbd);
  const ffnConH = dim(nEmbd * 4);
  addVolumeStack(group, FFN_CON_X, LOW_Y, BACK_Z + 0.1, ffnConW, ffnConH, 0.95, 0xf472b6, 2);

  addRod(group, [LN1_X + 0.5, TOP_Y, MID_Z], [MHA_X - 0.8, TOP_Y - 0.1, FRONT_Z - 0.15], 0x3cb6df, 0.03, 0.3);
  addRod(group, [MHA_X + 0.95, MID_Y, MID_Z], [PROJ_X - 0.8, MID_Y, MID_Z], 0x3cb6df, 0.03, 0.3);
  addRod(group, [PROJ_X + 0.85, MID_Y, MID_Z], [LN2_X - 0.7, MID_Y, MID_Z], 0x3cb6df, 0.03, 0.3);
  addRod(group, [LN2_X + 0.7, MID_Y, MID_Z], [FFN_EXP_X - 0.95, LOW_Y, BACK_Z], 0x3cb6df, 0.03, 0.3);
  addRod(group, [FFN_EXP_X + 0.9, LOW_Y, BACK_Z], [FFN_CON_X - 0.6, LOW_Y, BACK_Z + 0.05], 0x3cb6df, 0.03, 0.3);

  addRod(group, [-7.45, 0, 0.45], [-7.45, SKIP_Y, 0.45], 0x4ecdc4, 0.04, 0.52);
  addRod(group, [-7.45, SKIP_Y, 0.45], [-0.45, SKIP_Y, 0.45], 0x4ecdc4, 0.04, 0.52);
  addRod(group, [-0.45, SKIP_Y, 0.45], [-0.45, 0, 0.45], 0x4ecdc4, 0.04, 0.52);

  addRod(group, [-0.2, 0, -0.45], [-0.2, SKIP_Y, -0.45], 0x4ecdc4, 0.04, 0.52);
  addRod(group, [-0.2, SKIP_Y, -0.45], [7.85, SKIP_Y, -0.45], 0x4ecdc4, 0.04, 0.52);
  addRod(group, [7.85, SKIP_Y, -0.45], [7.85, 0, -0.45], 0x4ecdc4, 0.04, 0.52);

  group.userData.inputAnchor  = new THREE.Vector3(-9, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3( 9, 0, 0);

  return group;
}
