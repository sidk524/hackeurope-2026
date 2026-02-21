import * as TSP from 'tensorspace';
import { getNetworkArchitecture } from './api.js';

function getClassLabels(count) {
  const n = Math.max(2, Math.min(100, count || 10));
  return Array.from({ length: n }, (_, i) => `${i}`);
}

function addConv(model, params) {
  model.add(new TSP.layers.Conv2d({
    filters: params?.outChannels ?? 8,
    kernelSize: params?.kernelSize ?? [3, 3],
    strides: params?.stride ?? [1, 1],
    padding: Array.isArray(params?.padding) && (params.padding[0] > 0 || params.padding[1] > 0) ? 'same' : 'valid',
  }));
}

function addPool(model, params) {
  model.add(new TSP.layers.Pooling2d({
    poolSize: params?.kernelSize ?? [2, 2],
    strides: params?.stride ?? [2, 2],
    padding: 'valid',
  }));
}

function addDense(model, units) {
  model.add(new TSP.layers.Dense({ units: Math.max(2, units || 32) }));
}

function clearAppContainer() {
  const app = document.querySelector('#app');
  app.innerHTML = '';
  app.style.width = '100vw';
  app.style.height = '100vh';
  app.style.overflow = 'hidden';
  return app;
}

function createTooltip() {
  const tooltip = document.createElement('div');
  tooltip.style.position = 'fixed';
  tooltip.style.pointerEvents = 'none';
  tooltip.style.zIndex = '9999';
  tooltip.style.background = 'rgba(12, 16, 28, 0.9)';
  tooltip.style.color = '#d8ebff';
  tooltip.style.border = '1px solid rgba(120, 170, 255, 0.45)';
  tooltip.style.borderRadius = '8px';
  tooltip.style.padding = '8px 10px';
  tooltip.style.font = '12px/1.35 system-ui, -apple-system, Segoe UI, Roboto, sans-serif';
  tooltip.style.maxWidth = '320px';
  tooltip.style.boxShadow = '0 8px 24px rgba(0,0,0,0.25)';
  tooltip.style.display = 'none';
  document.body.appendChild(tooltip);
  return tooltip;
}

function paramsHtml(params) {
  if (!params || typeof params !== 'object') return '';
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== false)
    .slice(0, 8)
    .map(([k, v]) => `<div><span style="opacity:.75">${k}</span>: <span>${Array.isArray(v) ? `[${v.join(', ')}]` : v}</span></div>`)
    .join('');
}

function attachTensorSpaceHoverTooltip(model, layerMeta) {
  const renderer = model?.modelRenderer;
  const canvas = renderer?.sceneArea;
  const raycaster = renderer?.raycaster;
  const camera = renderer?.camera;
  const scene = renderer?.scene;
  if (!canvas || !raycaster || !camera || !scene) return;

  const tooltip = createTooltip();
  let activeKey = '';

  function hide() {
    tooltip.style.display = 'none';
    activeKey = '';
  }

  function resolveMeta(layerIndex) {
    if (!Number.isInteger(layerIndex)) return null;
    return layerMeta[layerIndex] || layerMeta[layerIndex - 1] || null;
  }

  canvas.addEventListener('mouseleave', hide);
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera({ x, y }, camera);
    const hits = raycaster.intersectObjects(scene.children, true);
    const hit = hits.find(h => Number.isInteger(h.object?.layerIndex));
    if (!hit) {
      hide();
      return;
    }

    const meta = resolveMeta(hit.object.layerIndex);
    if (!meta) {
      hide();
      return;
    }

    const key = `${meta.id}|${meta.type}|${JSON.stringify(meta.params || {})}`;
    if (key !== activeKey) {
      tooltip.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px">${meta.id}</div>
        <div style="opacity:.85;margin-bottom:4px">type: ${meta.type}</div>
        ${paramsHtml(meta.params)}
      `;
      activeKey = key;
    }

    tooltip.style.display = 'block';
    tooltip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 340)}px`;
    tooltip.style.top = `${Math.min(e.clientY + 14, window.innerHeight - 120)}px`;
  });
}

async function renderTensorSpaceFromReport() {
  const app = clearAppContainer();
  const network = await getNetworkArchitecture();
  const layers = Array.isArray(network?.layers) ? network.layers : [];
  const model = new TSP.models.Sequential(app, { stats: false, animeTime: 220 });
  const layerMeta = [];
  model.add(new TSP.layers.GreyscaleInput({ shape: [28, 28] }));
  layerMeta.push({ id: 'input', type: 'GreyscaleInput', params: { shape: [28, 28] } });

  const linearIndices = layers
    .map((layer, idx) => ({ idx, layer }))
    .filter(({ layer }) => layer?.type === 'Linear')
    .map(({ idx }) => idx);
  const lastLinearIdx = linearIndices.length ? linearIndices[linearIndices.length - 1] : -1;

  let hasSpatialLayer = false;
  let flattenAdded = false;
  let added = 0;

  for (let i = 0; i < layers.length; i++) {
    const layer = layers[i];
    if (!layer || !layer.type) continue;
    if (layer.type === 'Conv2d') {
      addConv(model, layer.params);
      layerMeta.push({ id: layer.id || `conv_${i}`, type: 'Conv2d', params: layer.params || {} });
      hasSpatialLayer = true;
      added += 1;
      continue;
    }
    if (layer.type === 'MaxPool2d') {
      addPool(model, layer.params);
      layerMeta.push({ id: layer.id || `pool_${i}`, type: 'MaxPool2d', params: layer.params || {} });
      hasSpatialLayer = true;
      added += 1;
      continue;
    }
    if (layer.type === 'Linear') {
      if (hasSpatialLayer && !flattenAdded) {
        model.add(new TSP.layers.Flatten({}));
        layerMeta.push({ id: 'flatten', type: 'Flatten', params: {} });
        flattenAdded = true;
      }
      const units = layer.params?.outFeatures ?? layer.params?.units ?? 32;
      if (i === lastLinearIdx) {
        model.add(new TSP.layers.Output1d({ outputs: getClassLabels(units) }));
        layerMeta.push({ id: layer.id || `output_${i}`, type: 'Output1d', params: { outputs: getClassLabels(units) } });
      } else {
        addDense(model, units);
        layerMeta.push({ id: layer.id || `dense_${i}`, type: 'Dense', params: { units } });
      }
      added += 1;
    }
  }

  if (added === 0) {
    model.add(new TSP.layers.Dense({ units: 64 }));
    layerMeta.push({ id: 'dense_fallback', type: 'Dense', params: { units: 64 } });
    model.add(new TSP.layers.Output1d({ outputs: getClassLabels(10) }));
    layerMeta.push({ id: 'output_fallback', type: 'Output1d', params: { outputs: getClassLabels(10) } });
  } else if (lastLinearIdx < 0) {
    model.add(new TSP.layers.Output1d({ outputs: getClassLabels(10) }));
    layerMeta.push({ id: 'output', type: 'Output1d', params: { outputs: getClassLabels(10) } });
  }

  model.init(() => {
    attachTensorSpaceHoverTooltip(model, layerMeta);
  });
}

renderTensorSpaceFromReport().catch((err) => {
  const app = clearAppContainer();
  const pre = document.createElement('pre');
  pre.textContent = `TensorSpace render failed:\n${err?.message || err}`;
  pre.style.padding = '16px';
  pre.style.whiteSpace = 'pre-wrap';
  app.appendChild(pre);
});
