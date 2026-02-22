"use client";

import { useEffect, useRef } from "react";
import type { Model } from "@/lib/client";

type ThreeSceneProps = {
  className?: string;
  model?: Model | null;
};

type NetworkLayer = {
  id: string;
  type: string;
  params?: Record<string, unknown>;
};

type ParsedNetwork = {
  name: string;
  layers: NetworkLayer[];
};

const CONTAINER_TYPES = new Set(["GPTLanguageModel", "SmallCNN", "Sequential", "ModuleList"]);

function toIntPair(value: unknown, fallback: [number, number]): [number, number] {
  if (Array.isArray(value) && value.length >= 2) {
    const a = Number(value[0]);
    const b = Number(value[1]);
    if (Number.isFinite(a) && Number.isFinite(b)) {
      return [Math.max(1, Math.round(a)), Math.max(1, Math.round(b))];
    }
  }
  return fallback;
}

function toFinite(value: unknown, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getClassLabels(count: number): string[] {
  const n = Math.max(2, Math.min(100, count || 10));
  return Array.from({ length: n }, (_, i) => `${i}`);
}

function maybeInt(v: number): number | null {
  const n = Math.round(v);
  return Number.isFinite(v) && Math.abs(v - n) < 1e-6 ? n : null;
}

function inferLinearDims(parameters: number | null, nEmbd: number | null) {
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

function flattenModuleTree(
  node: unknown,
  path: string,
  out: NetworkLayer[],
  flatLayers: Record<string, { type?: string; parameters?: number }>
) {
  if (!node || typeof node !== "object") return;
  const rec = node as { type?: string; children?: Record<string, unknown> };
  const type = rec.type || "Unknown";
  const isContainer = CONTAINER_TYPES.has(type);
  if (!isContainer) {
    const info = flatLayers[path] || {};
    out.push({
      id: path || type,
      type,
      params: { parameters: info.parameters ?? null },
    });
  }
  if (!rec.children) return;
  for (const [name, child] of Object.entries(rec.children)) {
    const childPath = path ? `${path}.${name}` : name;
    flattenModuleTree(child, childPath, out, flatLayers);
  }
}

function parseModel(model: Model | null | undefined): ParsedNetwork {
  const architecture = (model?.architecture ?? {}) as Record<string, unknown>;
  const hyperparameters = (model?.hyperparameters ?? {}) as Record<string, unknown>;
  const runName = typeof architecture.name === "string" ? architecture.name : "Neural Network";

  const layerList = Array.isArray(architecture.layers) ? architecture.layers : [];
  if (layerList.length > 0) {
    const normalized = layerList
      .map<NetworkLayer | null>((layer, idx) => {
        if (!layer || typeof layer !== "object") return null;
        const rec = layer as Record<string, unknown>;
        const type = typeof rec.type === "string" ? rec.type : null;
        if (!type) return null;
        return {
          id: typeof rec.id === "string" ? rec.id : `layer_${idx}`,
          type,
          params:
            rec.params && typeof rec.params === "object"
              ? (rec.params as Record<string, unknown>)
              : {},
        };
      })
      .filter((v): v is NetworkLayer => v !== null);
    return { name: runName, layers: normalized };
  }

  const modelArchitecture =
    architecture.model_architecture && typeof architecture.model_architecture === "object"
      ? (architecture.model_architecture as Record<string, unknown>)
      : architecture;
  const flatLayers =
    modelArchitecture.layers && typeof modelArchitecture.layers === "object"
      ? (modelArchitecture.layers as Record<string, { type?: string; parameters?: number }>)
      : {};
  const tree =
    modelArchitecture.module_tree && typeof modelArchitecture.module_tree === "object"
      ? modelArchitecture.module_tree
      : null;
  const nEmbd = toFinite(hyperparameters.n_embd, NaN);
  const nHead = toFinite(hyperparameters.n_head, NaN);
  const tokenEmbParams =
    flatLayers.token_embedding_table && Number.isFinite(flatLayers.token_embedding_table.parameters)
      ? Number(flatLayers.token_embedding_table.parameters)
      : NaN;
  const vocabSize =
    Number.isFinite(tokenEmbParams) && Number.isFinite(nEmbd) && nEmbd > 0
      ? Math.round(tokenEmbParams / nEmbd)
      : null;
  const nn = Number.isFinite(nEmbd) ? nEmbd : null;
  const nh = Number.isFinite(nHead) && nHead > 0 ? nHead : 1;
  const headSize = nn ? Math.max(1, Math.round(nn / nh)) : 8;
  const layers: NetworkLayer[] = [];
  if (tree) {
    flattenModuleTree(tree, "", layers, flatLayers);
  } else {
    for (const [id, info] of Object.entries(flatLayers)) {
      layers.push({
        id,
        type: info?.type || "Unknown",
        params: { parameters: info?.parameters ?? null },
      });
    }
  }

  const enriched = layers.map((layer) => {
    const params = layer.params ?? {};
    const parameters = toFinite(params.parameters, NaN);
    if (layer.type === "Linear") {
      return {
        ...layer,
        params: {
          ...params,
          ...inferLinearDims(Number.isFinite(parameters) ? parameters : null, nn),
          outFeatures:
            typeof params.outFeatures === "number"
              ? params.outFeatures
              : typeof params.units === "number"
                ? params.units
                : inferLinearDims(Number.isFinite(parameters) ? parameters : null, nn).outFeatures,
        },
      };
    }
    if (layer.type === "Conv2d") {
      return {
        ...layer,
        params: {
          ...params,
          outChannels: toFinite(params.outChannels, Math.max(4, Math.round(Math.sqrt(parameters || 64)))),
          kernelSize: toIntPair(params.kernelSize, [3, 3]),
          stride: toIntPair(params.stride, [1, 1]),
          padding: toIntPair(params.padding, [0, 0]),
        },
      };
    }
    if (layer.type === "MaxPool2d") {
      return {
        ...layer,
        params: {
          ...params,
          kernelSize: toIntPair(params.kernelSize, [2, 2]),
          stride: toIntPair(params.stride, [2, 2]),
        },
      };
    }
    if (layer.type === "Head") {
      return {
        ...layer,
        params: { ...params, headSize },
      };
    }
    if (layer.type === "Embedding") {
      return {
        ...layer,
        params: {
          ...params,
          embeddingDim: toFinite(params.embeddingDim, nn ?? 8),
          vocabSize: toFinite(params.vocabSize, vocabSize ?? 32),
        },
      };
    }
    return layer;
  });

  return { name: runName, layers: enriched };
}

export default function ThreeScene({ className = "", model }: ThreeSceneProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let disposed = false;
    let modelRenderer: { animate?: () => void; stopAnimate?: () => void } | null = null;
    let canvas: HTMLElement | null = null;
    let onMove: ((e: MouseEvent) => void) | null = null;
    let onLeave: (() => void) | null = null;
    container.innerHTML = "";

    const run = async () => {
      try {
        const parsed = parseModel(model);
        const TSP = await import("tensorspace");
        if (disposed) return;
        const seq = new TSP.models.Sequential(container, { stats: false, animeTime: 220 });
        modelRenderer = seq?.modelRenderer ?? null;
        seq.add(new TSP.layers.GreyscaleInput({ shape: [28, 28] }));

        const layers = parsed.layers;
        const linearIndices = layers
          .map((layer, idx) => ({ layer, idx }))
          .filter(({ layer }) => layer.type === "Linear")
          .map(({ idx }) => idx);
        const lastLinearIdx = linearIndices.length ? linearIndices[linearIndices.length - 1] : -1;
        let hasSpatialLayer = false;
        let flattenAdded = false;
        let added = 0;
        const layerMeta: Array<{ id: string; type: string; params: Record<string, unknown> }> = [
          { id: "input", type: "GreyscaleInput", params: { shape: [28, 28] } },
        ];

        for (let i = 0; i < layers.length; i++) {
          const layer = layers[i];
          const params = layer.params ?? {};
          if (layer.type === "Conv2d") {
            const outChannels = Math.max(1, Math.round(toFinite(params.outChannels, 8)));
            const kernelSize = toIntPair(params.kernelSize, [3, 3]);
            const stride = toIntPair(params.stride, [1, 1]);
            const padding = toIntPair(params.padding, [0, 0]);
            seq.add(
              new TSP.layers.Conv2d({
                filters: outChannels,
                kernelSize,
                strides: stride,
                padding: padding[0] > 0 || padding[1] > 0 ? "same" : "valid",
              })
            );
            layerMeta.push({ id: layer.id || `conv_${i}`, type: "Conv2d", params });
            hasSpatialLayer = true;
            added += 1;
            continue;
          }
          if (layer.type === "MaxPool2d") {
            const kernelSize = toIntPair(params.kernelSize, [2, 2]);
            const stride = toIntPair(params.stride, [2, 2]);
            seq.add(
              new TSP.layers.Pooling2d({
                poolSize: kernelSize,
                strides: stride,
                padding: "valid",
              })
            );
            layerMeta.push({ id: layer.id || `pool_${i}`, type: "MaxPool2d", params });
            hasSpatialLayer = true;
            added += 1;
            continue;
          }
          if (layer.type === "Linear") {
            if (hasSpatialLayer && !flattenAdded) {
              seq.add(new TSP.layers.Flatten({}));
              layerMeta.push({ id: "flatten", type: "Flatten", params: {} });
              flattenAdded = true;
            }
            const units = Math.max(
              2,
              Math.round(toFinite(params.outFeatures, toFinite(params.units, 32)))
            );
            if (i === lastLinearIdx) {
              seq.add(new TSP.layers.Output1d({ units, outputs: getClassLabels(units) }));
              layerMeta.push({
                id: layer.id || `output_${i}`,
                type: "Output1d",
                params: { units, outputs: getClassLabels(units) },
              });
            } else {
              seq.add(new TSP.layers.Dense({ units }));
              layerMeta.push({
                id: layer.id || `dense_${i}`,
                type: "Dense",
                params: { units },
              });
            }
            added += 1;
            continue;
          }
        }

        if (added === 0) {
          seq.add(new TSP.layers.Dense({ units: 64 }));
          seq.add(new TSP.layers.Output1d({ units: 10, outputs: getClassLabels(10) }));
          layerMeta.push({ id: "dense_fallback", type: "Dense", params: { units: 64 } });
          layerMeta.push({
            id: "output_fallback",
            type: "Output1d",
            params: { units: 10, outputs: getClassLabels(10) },
          });
        } else if (lastLinearIdx < 0) {
          seq.add(new TSP.layers.Output1d({ units: 10, outputs: getClassLabels(10) }));
          layerMeta.push({ id: "output", type: "Output1d", params: { units: 10, outputs: getClassLabels(10) } });
        }

        const tooltip = document.createElement("div");
        tooltip.style.position = "fixed";
        tooltip.style.pointerEvents = "none";
        tooltip.style.zIndex = "9999";
        tooltip.style.background = "rgba(12, 16, 28, 0.9)";
        tooltip.style.color = "#d8ebff";
        tooltip.style.border = "1px solid rgba(120, 170, 255, 0.45)";
        tooltip.style.borderRadius = "8px";
        tooltip.style.padding = "8px 10px";
        tooltip.style.font = "12px/1.35 system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
        tooltip.style.maxWidth = "320px";
        tooltip.style.boxShadow = "0 8px 24px rgba(0,0,0,0.25)";
        tooltip.style.display = "none";
        document.body.appendChild(tooltip);
        tooltipRef.current = tooltip;

        seq.init(() => {
          const renderer = seq?.modelRenderer as
            | {
                sceneArea?: HTMLElement;
                raycaster?: { setFromCamera: (p: { x: number; y: number }, c: unknown) => void; intersectObjects: (o: unknown[], r: boolean) => Array<{ object?: { layerIndex?: number } }> };
                camera?: unknown;
                scene?: { children?: unknown[] };
              }
            | undefined;
          canvas = renderer?.sceneArea ?? null;
          const raycaster = renderer?.raycaster;
          const camera = renderer?.camera;
          const scene = renderer?.scene as { rotation?: { z: number }; children?: unknown[] } | undefined;
          const sceneChildren = scene?.children;
          if (!canvas || !raycaster || !camera || !sceneChildren) return;
          // Rotate scene so the network flows sideways (left–right) instead of upwards
          if (scene?.rotation) scene.rotation.z = Math.PI / 2;
          let activeKey = "";
          const hide = () => {
            if (!tooltipRef.current) return;
            tooltipRef.current.style.display = "none";
            activeKey = "";
          };
          onLeave = hide;
          onMove = (e: MouseEvent) => {
            const rect = canvas!.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            raycaster.setFromCamera({ x, y }, camera);
            const hits = raycaster.intersectObjects(sceneChildren, true);
            const hit = hits.find((h) => Number.isInteger(h.object?.layerIndex));
            if (!hit) {
              hide();
              return;
            }
            const rawLayerIndex = hit.object?.layerIndex;
            const layerIndex = Number.isInteger(rawLayerIndex) ? Number(rawLayerIndex) : null;
            if (layerIndex == null) {
              hide();
              return;
            }
            const meta = layerMeta[layerIndex] || layerMeta[layerIndex - 1] || null;
            if (!meta) {
              hide();
              return;
            }
            const key = `${meta.id}|${meta.type}|${JSON.stringify(meta.params || {})}`;
            if (key !== activeKey && tooltipRef.current) {
              const paramsLines = Object.entries(meta.params || {})
                .filter(([, v]) => v !== undefined && v !== null && v !== false)
                .slice(0, 8)
                .map(
                  ([k, v]) =>
                    `<div><span style="opacity:.75">${k}</span>: <span>${Array.isArray(v) ? `[${v.join(", ")}]` : String(v)}</span></div>`
                )
                .join("");
              tooltipRef.current.innerHTML = `<div style="font-weight:600;margin-bottom:4px">${meta.id}</div><div style="opacity:.85;margin-bottom:4px">type: ${meta.type}</div>${paramsLines}`;
              activeKey = key;
            }
            if (tooltipRef.current) {
              tooltipRef.current.style.display = "block";
              tooltipRef.current.style.left = `${Math.min(e.clientX + 14, window.innerWidth - 340)}px`;
              tooltipRef.current.style.top = `${Math.min(e.clientY + 14, window.innerHeight - 120)}px`;
            }
          };
          canvas.addEventListener("mouseleave", onLeave);
          canvas.addEventListener("mousemove", onMove);
        });
      } catch {
        if (disposed) return;
        container.innerHTML =
          '<div class="flex h-full items-center justify-center text-sm text-zinc-500">Visualization unavailable</div>';
      }
    };
    void run();

    return () => {
      disposed = true;
      if (canvas && onLeave) canvas.removeEventListener("mouseleave", onLeave);
      if (canvas && onMove) canvas.removeEventListener("mousemove", onMove);
      if (tooltipRef.current && tooltipRef.current.parentNode) {
        tooltipRef.current.parentNode.removeChild(tooltipRef.current);
      }
      tooltipRef.current = null;
      if (modelRenderer?.stopAnimate) modelRenderer.stopAnimate();
      if (containerRef.current) containerRef.current.innerHTML = "";
    };
  }, [model]);

  return (
    <div
      className={`rounded-3xl border border-zinc-800 bg-zinc-950/60 p-4 shadow-lg ${className}`}
    >
      <div
        ref={containerRef}
        className="h-[420px] w-full rounded-2xl bg-zinc-900"
      />
    </div>
  );
}
