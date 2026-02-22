"use client";

import { useEffect, useRef, useState } from "react";
import type { Model } from "@/lib/client";

type ThreeSceneProps = {
  className?: string;
  model?: Model | null;
  sustainabilityScores?: Record<string, number>;
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

type LayerMetaItem = {
  id: string;
  type: string;
  params: Record<string, unknown>;
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

function scoreToColor(score: number): string {
  const clamped = Math.max(0, Math.min(100, score));
  const hue = (clamped / 100) * 120;
  return `hsl(${hue}, 85%, 48%)`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function tintMaterial(material: unknown, color: string): void {
  const rec = asRecord(material);
  const colorRef = asRecord(rec.color);
  const setStyle = colorRef.setStyle;
  if (typeof setStyle === "function") {
    setStyle.call(rec.color, color);
  }
}

function tintObjectTree(
  node: unknown,
  layerMeta: LayerMetaItem[],
  sustainabilityScores: Record<string, number>
): void {
  const rec = asRecord(node);
  const rawLayerIndex = rec.layerIndex;
  if (Number.isInteger(rawLayerIndex)) {
    const idx = Number(rawLayerIndex);
    const meta = layerMeta[idx] ?? layerMeta[idx - 1];
    if (meta) {
      const score = sustainabilityScores[meta.id];
      if (typeof score === "number") {
        const color = scoreToColor(score);
        const material = rec.material;
        if (Array.isArray(material)) {
          for (const m of material) tintMaterial(m, color);
        } else {
          tintMaterial(material, color);
        }
      }
    }
  }
  const children = rec.children;
  if (Array.isArray(children)) {
    for (const child of children) {
      tintObjectTree(child, layerMeta, sustainabilityScores);
    }
  }
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

export default function ThreeScene({ className = "", model, sustainabilityScores = {} }: ThreeSceneProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayContainerRef = useRef<HTMLDivElement | null>(null);
  const tooltipRef = useRef<HTMLDivElement | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  useEffect(() => {
    if (!isExpanded) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setIsExpanded(false);
    };
    window.addEventListener("keydown", onKeyDown);
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = previousOverflow;
    };
  }, [isExpanded]);

  useEffect(() => {
    const container = isExpanded ? overlayContainerRef.current : containerRef.current;
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
        const layerMeta: LayerMetaItem[] = [
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
          for (const child of sceneChildren) {
            tintObjectTree(child, layerMeta, sustainabilityScores);
          }
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
              const scoreValue = sustainabilityScores[meta.id];
              const sustainabilityLine =
                typeof scoreValue === "number"
                  ? `<div style="opacity:.95;margin-bottom:4px">sustainability: <span style="font-weight:600">${Math.max(0, Math.min(100, Math.round(scoreValue)))}/100</span></div>`
                  : "";
              tooltipRef.current.innerHTML = `<div style="font-weight:600;margin-bottom:4px">${meta.id}</div><div style="opacity:.85;margin-bottom:4px">type: ${meta.type}</div>${sustainabilityLine}${paramsLines}`;
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
  }, [model, isExpanded, sustainabilityScores]);

  return (
    <>
      <div
        className={`relative rounded-3xl border border-zinc-800 bg-zinc-950/60 p-4 shadow-lg ${className}`}
      >
        <button
          type="button"
          onClick={() => setIsExpanded(true)}
          aria-label="Expand visualization"
          className="absolute right-6 top-6 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1 text-xs text-zinc-200 transition hover:border-zinc-500 hover:bg-zinc-800"
        >
          Expand
        </button>
        <div
          ref={containerRef}
          className="h-[420px] w-full rounded-2xl bg-zinc-900"
        />
      </div>
      {isExpanded ? (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-6">
          <div className="relative w-full max-w-7xl rounded-3xl border border-zinc-700 bg-zinc-950 p-5 shadow-2xl">
            <button
              type="button"
              onClick={() => setIsExpanded(false)}
              aria-label="Close expanded visualization"
              className="absolute right-5 top-5 rounded-full border border-zinc-700 bg-zinc-900/80 px-3 py-1 text-xs text-zinc-200 transition hover:border-zinc-500 hover:bg-zinc-800"
            >
              Close
            </button>
            <div
              ref={overlayContainerRef}
              className="h-[78vh] w-full rounded-2xl bg-zinc-900"
            />
          </div>
        </div>
      ) : null}
    </>
  );
}
