import * as THREE from 'three';

function mat(color, opacity = 0.9, glow = 0.08) {
  return new THREE.MeshPhongMaterial({
    color,
    emissive: new THREE.Color(color),
    emissiveIntensity: glow,
    transparent: true,
    opacity,
    shininess: 42,
  });
}

function hash(i, r, c) {
  return (i * 37 + r * 19 + c * 11) % 23;
}

function voxelHeight(i, r, c, base, span) {
  return base + (((i * 13 + r * 7 + c * 5) % 9) / 8) * span;
}

export function createConv2d({
  outChannels = 16,
} = {}) {
  const group = new THREE.Group();
  const count = Math.max(1, Math.min(64, outChannels));
  const cols = Math.min(8, count);
  const rows = Math.ceil(count / cols);
  const tileSize = count > 32 ? 0.58 : 0.68;
  const gap = tileSize * 0.22;
  const tileDepth = 0.12;
  const voxelDepth = 0.08;
  const resolution = count > 32 ? 7 : 8;
  const voxelStep = tileSize / resolution;

  const totalX = cols * tileSize + (cols - 1) * gap;
  const totalY = rows * tileSize + (rows - 1) * gap;
  const startX = -totalX / 2 + tileSize / 2;
  const startY = -totalY / 2 + tileSize / 2;

  const backGeo = new THREE.BoxGeometry(tileSize * 1.04, tileSize * 1.04, tileDepth * 0.42);
  const plateGeo = new THREE.BoxGeometry(tileSize, tileSize, tileDepth);
  const voxelGeo = new THREE.BoxGeometry(voxelStep * 0.84, voxelStep * 0.84, voxelDepth);
  const backMat = mat(0x262c70, 0.72, 0.03);
  const plateMat = mat(0x6f63ff, 0.9, 0.06);
  const voxelMat = mat(0x3dd5ff, 0.95, 0.18);

  for (let i = 0; i < count; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = startX + col * (tileSize + gap);
    const y = startY + row * (tileSize + gap);

    const depthOffset = -0.04 - row * 0.009 + col * 0.003;
    const back = new THREE.Mesh(backGeo, backMat);
    back.position.set(x, y, depthOffset - tileDepth * 0.34);
    group.add(back);

    const plateDepthLocal = tileDepth * (1 + (i % 4) * 0.08);
    const plate = new THREE.Mesh(plateGeo, plateMat);
    plate.scale.z = plateDepthLocal / tileDepth;
    plate.position.set(x, y, depthOffset);
    group.add(plate);
    const plateTop = plate.position.z + plateDepthLocal / 2;

    const vStartX = x - tileSize / 2 + voxelStep / 2;
    const vStartY = y - tileSize / 2 + voxelStep / 2;
    for (let r = 0; r < resolution; r++) {
      for (let c = 0; c < resolution; c++) {
        if (hash(i, r, c) < 11) continue;
        const h = voxelHeight(i, r, c, voxelDepth * 0.85, voxelDepth * 1.25);
        const voxel = new THREE.Mesh(voxelGeo, voxelMat);
        voxel.scale.z = h / voxelDepth;
        voxel.position.set(
          vStartX + c * voxelStep,
          vStartY + r * voxelStep,
          plateTop + h / 2,
        );
        group.add(voxel);
      }
    }
  }

  group.userData.inputAnchor = new THREE.Vector3(-0.32, 0, 0);
  group.userData.outputAnchor = new THREE.Vector3(0.32, 0, 0);

  return group;
}
