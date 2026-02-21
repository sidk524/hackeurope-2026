import mockReport from '../cnn-mock-report.json';

const REMOTE_URL = 'http://192.168.1.100:5000';
const FETCH_TIMEOUT_MS = 1500;

const CONTAINER_TYPES = new Set([
  'GPTLanguageModel',
  'SmallCNN',
  'Sequential',
  'ModuleList',
]);

function isObserverReport(data) {
  return data && data.session && data.model_architecture;
}

function maybeInt(v) {
  const n = Math.round(v);
  return Number.isFinite(v) && Math.abs(v - n) < 1e-6 ? n : null;
}

function inferLinearDims(parameters, nEmbd) {
  if (!parameters || !nEmbd) {
    const s = Math.max(2, Math.round(Math.sqrt(parameters || 16)));
    return { inFeatures: s, outFeatures: s, showBias: false };
  }
  const noBiasOut = maybeInt(parameters / nEmbd);
  const biasOut = maybeInt(parameters / (nEmbd + 1));
  if (biasOut && biasOut > 0) {
    return { inFeatures: nEmbd, outFeatures: biasOut, showBias: true };
  }
  if (noBiasOut && noBiasOut > 0) {
    return { inFeatures: nEmbd, outFeatures: noBiasOut, showBias: false };
  }
  return {
    inFeatures: nEmbd,
    outFeatures: Math.max(2, Math.round(parameters / nEmbd)),
    showBias: false,
  };
}

function paramsFor(path, type, ctx) {
  const info = ctx.flatLayers[path] || {};
  const graphNode = ctx.layerGraphNodeById[path] || null;
  const parameters = info.parameters;
  const nEmbd = ctx.nEmbd || 8;
  const nHead = ctx.nHead || 1;
  const headSize = Math.max(1, Math.round(nEmbd / nHead));
  const blockSize = ctx.blockSize || 8;
  const dropRate = ctx.dropRate ?? 0.2;
  const vocabSize = ctx.vocabSize || Math.max(8, blockSize);

  if (type === 'Embedding') {
    if (path === 'token_embedding_table') return { vocabSize, embeddingDim: nEmbd };
    if (path === 'position_embedding_table') return { vocabSize: blockSize, embeddingDim: nEmbd };
    if (parameters && nEmbd) return { vocabSize: Math.max(2, Math.round(parameters / nEmbd)), embeddingDim: nEmbd };
    return { vocabSize, embeddingDim: nEmbd };
  }

  if (type === 'Linear') {
    if (path.endsWith('.key') || path.endsWith('.query') || path.endsWith('.value')) {
      return { inFeatures: nEmbd, outFeatures: headSize, showBias: false };
    }
    if (path.endsWith('.sa.proj')) {
      return { inFeatures: nEmbd, outFeatures: nEmbd, showBias: true };
    }
    if (path.includes('.ffwd.net.0')) {
      return { inFeatures: nEmbd, outFeatures: nEmbd * 4, showBias: true };
    }
    if (path.includes('.ffwd.net.2')) {
      return { inFeatures: nEmbd * 4, outFeatures: nEmbd, showBias: true };
    }
    if (path === 'lm_head') {
      return { inFeatures: nEmbd, outFeatures: vocabSize, showBias: false };
    }
    return inferLinearDims(parameters, nEmbd);
  }

  if (type === 'LayerNorm') {
    if (parameters) return { normalizedDim: Math.max(2, Math.round(parameters / 2)) };
    return { normalizedDim: nEmbd };
  }

  if (type === 'Dropout') {
    const units = path.includes('.ffwd.') ? nEmbd * 4 : nEmbd;
    return { units, dropRate };
  }

  if (type === 'ReLU') {
    const units = path.includes('.ffwd.') ? nEmbd * 4 : nEmbd;
    return { units };
  }

  if (type === 'Head') {
    return { headSize, blockSize };
  }

  if (type === 'MultiHeadAttention') {
    return { numHeads: nHead, headSize };
  }

  if (type === 'FeedFoward' || type === 'FeedForward') {
    return { nEmbd, expansion: 4 };
  }

  if (type === 'Block') {
    return { nEmbd, numHeads: nHead };
  }

  if (type === 'Conv2d') {
    return {
      inChannels: graphNode?.in_channels ?? 1,
      outChannels: graphNode?.out_channels ?? Math.max(4, Math.round(Math.sqrt(parameters || 64))),
      kernelSize: graphNode?.kernel_size ?? [3, 3],
      stride: graphNode?.stride ?? [1, 1],
      padding: graphNode?.padding ?? [0, 0],
      hasBias: graphNode?.has_bias ?? true,
    };
  }

  if (type === 'MaxPool2d') {
    const inferredChannels = graphNode?.in_channels ?? graphNode?.out_channels ?? ctx.prevOutChannelsById[path] ?? 16;
    return {
      kernelSize: graphNode?.kernel_size ?? [2, 2],
      stride: graphNode?.stride ?? [2, 2],
      channels: inferredChannels,
    };
  }

  return {
    size: parameters
      ? Math.max(1, Math.round(Math.sqrt(parameters)))
      : nEmbd,
  };
}

function flattenModuleTree(node, path, out, ctx, parentRenderableId = null) {
  if (!node) return;
  const type = node.type || 'Unknown';
  const isContainer = CONTAINER_TYPES.has(type);
  let currentRenderableId = parentRenderableId;
  if (!isContainer) {
    currentRenderableId = path || type;
    out.push({
      id: currentRenderableId,
      type,
      params: paramsFor(path, type, ctx),
      parentId: parentRenderableId,
    });
  }
  if (!node.children) return;
  Object.entries(node.children).forEach(([name, child]) => {
    const childPath = path ? `${path}.${name}` : name;
    flattenModuleTree(child, childPath, out, ctx, currentRenderableId);
  });
}

function parseObserverReport(report) {
  const name = report.session?.run_name ?? report.session?.run_id ?? 'Neural Network';
  const flatLayers = report.model_architecture?.layers ?? {};
  const tree = report.model_architecture?.module_tree ?? null;
  const hp = report.hyperparameters ?? {};
  const nEmbd = hp.n_embd ?? null;
  const nHead = hp.n_head ?? null;
  const blockSize = hp.block_size ?? null;
  const dropRate = hp.dropout ?? null;
  const tokenEmbParams = flatLayers.token_embedding_table?.parameters ?? null;
  const vocabSize = tokenEmbParams && nEmbd ? Math.round(tokenEmbParams / nEmbd) : null;
  const layerGraphNodes = report.model_architecture?.layer_graph?.nodes ?? [];
  const layerGraphNodeById = Object.fromEntries(layerGraphNodes.map(node => [node.id, node]));
  const sequentialPath = report.model_architecture?.layer_graph?.sequential_path ?? [];
  const prevOutChannelsById = {};
  for (let i = 0; i < sequentialPath.length; i++) {
    const id = sequentialPath[i];
    const prev = i > 0 ? layerGraphNodeById[sequentialPath[i - 1]] : null;
    const prevOut = prev?.out_channels ?? prev?.in_channels ?? null;
    if (prevOut) prevOutChannelsById[id] = prevOut;
  }

  const ctx = { flatLayers, nEmbd, nHead, blockSize, dropRate, vocabSize, layerGraphNodeById, prevOutChannelsById };
  const layers = [];

  if (tree) {
    flattenModuleTree(tree, '', layers, ctx);
  } else {
    Object.entries(flatLayers).forEach(([id, info]) => {
      const type = info.type || 'Unknown';
      layers.push({
        id,
        type,
        params: paramsFor(id, type, ctx),
        parentId: null,
      });
    });
  }

  const childCountByParentId = {};
  layers.forEach((layer) => {
    if (!layer.parentId) return;
    childCountByParentId[layer.parentId] = (childCountByParentId[layer.parentId] || 0) + 1;
  });
  layers.forEach((layer) => {
    layer.hasChildren = Boolean(childCountByParentId[layer.id]);
  });

  return { name, layers };
}

function normalizeNetworkShape(data) {
  const name = data?.name || data?.session?.run_name || data?.session?.run_id || 'Neural Network';
  const rawLayers = Array.isArray(data?.layers) ? data.layers : [];
  const layers = rawLayers
    .map((layer, idx) => {
      if (!layer || typeof layer !== 'object') return null;
      const id = layer.id || `layer_${idx}`;
      if (layer.type) {
        return {
          id,
          type: layer.type,
          params: layer.params || {},
          parentId: layer.parentId || null,
          hasChildren: Boolean(layer.hasChildren),
        };
      }
      if (layer.size !== undefined) {
        return { id, size: layer.size };
      }
      return null;
    })
    .filter(Boolean);
  return { name, layers };
}

function fallbackNetwork() {
  if (isObserverReport(mockReport)) return normalizeNetworkShape(parseObserverReport(mockReport));
  return normalizeNetworkShape(mockReport);
}

export async function getNetworkArchitecture() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(`${REMOTE_URL}/network/architecture`, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const parsed = isObserverReport(data) ? parseObserverReport(data) : data;
    const normalized = normalizeNetworkShape(parsed);
    if (!normalized.layers.length) throw new Error('Parsed architecture has no renderable layers');
    clearTimeout(timeoutId);
    return normalized;
  } catch (err) {
    clearTimeout(timeoutId);
    console.warn('Could not reach remote server, using mock data.', err);
    return fallbackNetwork();
  }
}
