"use client";

import type { TrainStep } from "@/lib/client";
import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

// ── Helpers ──────────────────────────────────────────────────────────────────

function safe(v: unknown): number | null {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function safeStr(v: unknown): string {
  if (v == null) return "-";
  return String(v);
}

function safeBool(v: unknown): boolean {
  return v === true;
}

function fmt(n: number | null | undefined, digits = 2): string {
  if (n == null) return "-";
  return n.toFixed(digits);
}

// ── Data transformation types ────────────────────────────────────────────────

type TimeSeriesPoint = {
  stepIndex: number;
  // Loss
  trainMean: number | null;
  trainMin: number | null;
  trainMax: number | null;
  trainStd: number | null;
  valLoss: number | null;
  valAcc: number | null;
  // Throughput
  samplesPerSec: number | null;
  tokensPerSec: number | null;
  batchesPerSec: number | null;
  // Memory
  processRssMb: number | null;
  cudaAllocatedMb: number | null;
  cudaPeakAllocatedMb: number | null;
  // System
  cpuPercent: number | null;
  ramPercent: number | null;
  // Duration
  durationSeconds: number | null;
};

type TopOp = {
  name: string;
  calls: number;
  cpuTimeMs: number;
  avgCpuUs: number;
};

type OpCategory = {
  name: string;
  cpuTimeMs: number;
  pctCpu: number;
};

type ProfilerSnapshot = {
  topOps: TopOp[];
  categories: OpCategory[];
  totalCpuMs: number;
  totalCudaMs: number;
  forwardMs: number;
  backwardMs: number;
  fwdBwdRatio: number;
  numUniqueOps: number;
};

type LayerHealthRow = {
  name: string;
  activationMean: number | null;
  activationStd: number | null;
  gradientNormMean: number | null;
  weightSparsity: number | null;
  isDead: boolean;
  hasVanishingGradients: boolean;
  hasFrozenOutput: boolean;
  hasNearZeroWeights: boolean;
  hasLowActivationVariance: boolean;
};

type ActivationCorrelation = {
  layerA: string;
  layerB: string;
  correlation: number;
};

type CarbonSnapshot = {
  epochCo2Kg: number | null;
  cumulativeCo2Kg: number | null;
  powerDrawWatts: number | null;
  cpuPowerW: number | null;
  gpuPowerW: number | null;
  ramPowerW: number | null;
  cpuEnergyKwh: number | null;
  gpuEnergyKwh: number | null;
  ramEnergyKwh: number | null;
  epochEnergyKwh: number | null;
  cpuUtilPct: number | null;
  gpuUtilPct: number | null;
  countryName: string;
  cpuModel: string;
  gpuModel: string;
};

// ── Data extraction ──────────────────────────────────────────────────────────

function buildTimeSeries(steps: TrainStep[]): TimeSeriesPoint[] {
  return steps.map((step) => {
    const loss = step.loss as Record<string, unknown> | undefined;
    const val = loss?.val as Record<string, unknown> | undefined;
    const tp = step.throughput as Record<string, unknown> | undefined;
    const mem = step.memory as Record<string, unknown> | undefined;
    const sys = step.system as Record<string, unknown> | undefined;

    return {
      stepIndex: step.step_index,
      trainMean: safe(loss?.train_mean),
      trainMin: safe(loss?.train_min),
      trainMax: safe(loss?.train_max),
      trainStd: safe(loss?.train_std),
      valLoss: safe(val?.val_loss),
      valAcc: safe(val?.val_acc) != null ? (safe(val?.val_acc) as number) * 100 : null,
      samplesPerSec: safe(tp?.samples_per_second),
      tokensPerSec: safe(tp?.tokens_per_second),
      batchesPerSec: safe(tp?.batches_per_second),
      processRssMb: safe(mem?.process_rss_mb),
      cudaAllocatedMb: safe(mem?.cuda_allocated_mb),
      cudaPeakAllocatedMb: safe(mem?.cuda_peak_allocated_mb),
      cpuPercent: safe(sys?.cpu_percent),
      ramPercent: safe(sys?.ram_percent),
      durationSeconds: safe(step.duration_seconds),
    };
  });
}

function extractProfiler(steps: TrainStep[]): ProfilerSnapshot | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    const p = steps[i].profiler as Record<string, unknown> | undefined;
    if (!p || !p.top_operations) continue;

    const rawOps = p.top_operations as Array<Record<string, unknown>>;
    const topOps: TopOp[] = rawOps.slice(0, 15).map((op) => ({
      name: safeStr(op.name),
      calls: safe(op.calls) ?? 0,
      cpuTimeMs: safe(op.cpu_time_ms) ?? 0,
      avgCpuUs: safe(op.avg_cpu_us) ?? 0,
    }));

    const rawCats = p.operation_categories as Record<string, Record<string, unknown>> | undefined;
    const categories: OpCategory[] = rawCats
      ? Object.entries(rawCats).map(([name, data]) => ({
          name: name.replace(/_/g, " "),
          cpuTimeMs: safe(data.cpu_time_ms) ?? 0,
          pctCpu: safe(data.pct_cpu) ?? 0,
        }))
      : [];

    return {
      topOps,
      categories: categories.sort((a, b) => b.pctCpu - a.pctCpu),
      totalCpuMs: safe(p.total_cpu_time_ms) ?? 0,
      totalCudaMs: safe(p.total_cuda_time_ms) ?? 0,
      forwardMs: safe(p.forward_time_ms) ?? 0,
      backwardMs: safe(p.backward_time_ms) ?? 0,
      fwdBwdRatio: safe(p.fwd_bwd_ratio) ?? 0,
      numUniqueOps: safe(p.num_unique_ops) ?? 0,
    };
  }
  return null;
}

function extractLayerHealth(
  steps: TrainStep[]
): { layers: LayerHealthRow[]; correlations: ActivationCorrelation[] } | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    const lh = steps[i].layer_health as Record<string, unknown> | undefined;
    if (!lh?.layers) continue;

    const rawLayers = lh.layers as Record<string, Record<string, unknown>>;
    const layers: LayerHealthRow[] = Object.entries(rawLayers).map(
      ([name, data]) => ({
        name,
        activationMean: safe(data.activation_mean),
        activationStd: safe(data.activation_std),
        gradientNormMean: safe(data.gradient_norm_mean),
        weightSparsity: safe(data.weight_sparsity),
        isDead: safeBool(data.is_dead),
        hasVanishingGradients: safeBool(data.has_vanishing_gradients),
        hasFrozenOutput: safeBool(data.has_frozen_output),
        hasNearZeroWeights: safeBool(data.has_near_zero_weights),
        hasLowActivationVariance: safeBool(data.has_low_activation_variance),
      })
    );

    const rawCorr = lh.activation_correlations as
      | Array<Record<string, unknown>>
      | undefined;
    const correlations: ActivationCorrelation[] = (rawCorr ?? []).map((c) => ({
      layerA: safeStr(c.layer_a),
      layerB: safeStr(c.layer_b),
      correlation: safe(c.correlation) ?? 0,
    }));

    return { layers, correlations };
  }
  return null;
}

function extractCarbon(steps: TrainStep[]): CarbonSnapshot | null {
  for (let i = steps.length - 1; i >= 0; i--) {
    const c = steps[i].carbon_emissions as Record<string, unknown> | undefined;
    if (!c) continue;

    return {
      epochCo2Kg: safe(c.epoch_co2_kg),
      cumulativeCo2Kg: safe(c.cumulative_co2_kg),
      powerDrawWatts: safe(c.power_draw_watts),
      cpuPowerW: safe(c.cpu_power_w),
      gpuPowerW: safe(c.gpu_power_w),
      ramPowerW: safe(c.ram_power_w),
      cpuEnergyKwh: safe(c.cpu_energy_kwh),
      gpuEnergyKwh: safe(c.gpu_energy_kwh),
      ramEnergyKwh: safe(c.ram_energy_kwh),
      epochEnergyKwh: safe(c.epoch_energy_kwh),
      cpuUtilPct: safe(c.cpu_utilization_pct),
      gpuUtilPct: safe(c.gpu_utilization_pct),
      countryName: safeStr(c.country_name),
      cpuModel: safeStr(c.cpu_model),
      gpuModel: safeStr(c.gpu_model),
    };
  }
  return null;
}

// ── Explain prompts ──────────────────────────────────────────────────────────

function lossExplainPrompt(ts: TimeSeriesPoint[]): string {
  const last = ts[ts.length - 1];
  return `Analyze the loss trend across ${ts.length} training steps. The latest train loss is ${fmt(last?.trainMean, 4)}, val loss is ${fmt(last?.valLoss, 4)}, val accuracy is ${fmt(last?.valAcc, 1)}%. Is the model converging? Are there any anomalies or signs of overfitting?`;
}

function throughputExplainPrompt(ts: TimeSeriesPoint[]): string {
  const last = ts[ts.length - 1];
  return `Analyze the throughput metrics across ${ts.length} training steps. Latest: ${fmt(last?.samplesPerSec, 1)} samples/s, ${fmt(last?.tokensPerSec, 1)} tokens/s. Are there throughput bottlenecks or degradation over time?`;
}

function memoryExplainPrompt(ts: TimeSeriesPoint[]): string {
  const last = ts[ts.length - 1];
  return `Analyze memory usage across ${ts.length} steps. Latest process RSS: ${fmt(last?.processRssMb, 1)} MB, CUDA allocated: ${fmt(last?.cudaAllocatedMb, 1)} MB. Is there a memory leak? Any concerns?`;
}

function systemExplainPrompt(ts: TimeSeriesPoint[]): string {
  const last = ts[ts.length - 1];
  return `Analyze system utilization across ${ts.length} steps. Latest CPU: ${fmt(last?.cpuPercent, 1)}%, RAM: ${fmt(last?.ramPercent, 1)}%. Is the hardware being utilized efficiently?`;
}

function profilerExplainPrompt(prof: ProfilerSnapshot): string {
  const topOp = prof.topOps[0];
  return `Analyze this profiler snapshot: forward/backward ratio = ${fmt(prof.fwdBwdRatio, 3)}, total CPU time = ${fmt(prof.totalCpuMs, 1)}ms, total CUDA time = ${fmt(prof.totalCudaMs, 1)}ms, ${prof.numUniqueOps} unique ops. Top operation is "${topOp?.name}" at ${fmt(topOp?.cpuTimeMs, 1)}ms. What are the performance bottlenecks and what can be optimized?`;
}

function layerHealthExplainPrompt(layers: LayerHealthRow[]): string {
  const deadCount = layers.filter((l) => l.isDead).length;
  const vanishingCount = layers.filter((l) => l.hasVanishingGradients).length;
  const frozenCount = layers.filter((l) => l.hasFrozenOutput).length;
  return `Analyze layer health: ${layers.length} layers total, ${deadCount} dead, ${vanishingCount} with vanishing gradients, ${frozenCount} with frozen output. What should I fix and in what priority order?`;
}

function carbonExplainPrompt(carbon: CarbonSnapshot): string {
  return `Analyze carbon emissions: ${fmt(carbon.epochCo2Kg, 6)} kg CO2 per epoch, ${fmt(carbon.powerDrawWatts, 1)}W power draw, CPU: ${fmt(carbon.cpuPowerW, 1)}W, GPU: ${fmt(carbon.gpuPowerW, 1)}W. Country: ${carbon.countryName}. How can I reduce the carbon footprint of this training run?`;
}

// ── Sub-components ───────────────────────────────────────────────────────────

function ExplainButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-1.5 rounded-lg border border-indigo-700/60 bg-indigo-950/40 px-2.5 py-1.5 text-xs text-indigo-300 transition hover:border-indigo-500 hover:bg-indigo-900/50 hover:text-indigo-100"
      title="Ask Atlas to explain this panel"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="12"
        height="12"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden
      >
        <circle cx="12" cy="12" r="10" />
        <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      Explain
    </button>
  );
}

function DashboardPanel({
  category,
  title,
  onExplain,
  badge,
  children,
}: {
  category: string;
  title: string;
  onExplain?: () => void;
  badge?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            {category}
          </p>
          <h2 className="text-lg font-semibold">{title}</h2>
        </div>
        <div className="flex items-center gap-2">
          {badge && (
            <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
              {badge}
            </span>
          )}
          {onExplain && <ExplainButton onClick={onExplain} />}
        </div>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function MetricCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-3 py-2.5">
      <p className="text-[10px] uppercase tracking-wider text-zinc-500">
        {label}
      </p>
      <p className="mt-1 text-sm font-semibold text-zinc-100">{value}</p>
      {sub && <p className="text-[10px] text-zinc-500">{sub}</p>}
    </div>
  );
}

function ChartTooltip({
  active,
  payload,
  label,
  fields,
}: {
  active?: boolean;
  payload?: Array<{
    name: string;
    value?: number;
    color?: string;
  }>;
  label?: string | number;
  fields?: string[];
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 shadow-xl">
      <div className="font-semibold text-zinc-400">Step {label}</div>
      {payload
        .filter((p) => p.value != null && (!fields || fields.includes(p.name)))
        .map((p, i) => (
          <div key={i} className="flex items-center gap-2">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ backgroundColor: p.color }}
            />
            <span className="text-zinc-400">{p.name}:</span>
            <span className="font-medium">{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</span>
          </div>
        ))}
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className="flex h-48 items-center justify-center rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/40 text-sm text-zinc-500">
      {message}
    </div>
  );
}

// ── Chart panels ─────────────────────────────────────────────────────────────

function LossTrendPanel({
  data,
  onExplain,
}: {
  data: TimeSeriesPoint[];
  onExplain?: () => void;
}) {
  const hasLoss = data.some((d) => d.trainMean != null);
  const hasValAcc = data.some((d) => d.valAcc != null);

  return (
    <DashboardPanel category="Loss" title="Training loss" onExplain={onExplain}>
      {!hasLoss ? (
        <EmptyState message="No loss data available" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="stepIndex"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                label={{
                  value: "Step",
                  position: "insideBottom",
                  offset: -4,
                  fill: "var(--muted-foreground)",
                  fontSize: 10,
                }}
              />
              <YAxis
                yAxisId="loss"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                domain={["auto", "auto"]}
              />
              {hasValAcc && (
                <YAxis
                  yAxisId="acc"
                  orientation="right"
                  tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                  tickLine={{ stroke: "var(--border)" }}
                  axisLine={{ stroke: "var(--border)" }}
                  domain={[0, 100]}
                />
              )}
              <Tooltip content={<ChartTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => (
                  <span className="text-zinc-300">{value}</span>
                )}
              />
              <Line
                type="monotone"
                dataKey="trainMean"
                name="Train loss"
                yAxisId="loss"
                stroke="var(--chart-1)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-1)", r: 2 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="valLoss"
                name="Val loss"
                yAxisId="loss"
                stroke="var(--chart-2)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-2)", r: 2 }}
                connectNulls
              />
              {hasValAcc && (
                <Line
                  type="monotone"
                  dataKey="valAcc"
                  name="Val acc %"
                  yAxisId="acc"
                  stroke="var(--chart-3)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-3)", r: 2 }}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function ThroughputPanel({
  data,
  onExplain,
}: {
  data: TimeSeriesPoint[];
  onExplain?: () => void;
}) {
  const hasData = data.some(
    (d) =>
      d.samplesPerSec != null ||
      d.tokensPerSec != null ||
      d.batchesPerSec != null
  );

  return (
    <DashboardPanel
      category="Performance"
      title="Throughput"
      onExplain={onExplain}
    >
      {!hasData ? (
        <EmptyState message="No throughput data available" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="stepIndex"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <YAxis
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => (
                  <span className="text-zinc-300">{value}</span>
                )}
              />
              <Line
                type="monotone"
                dataKey="samplesPerSec"
                name="Samples/s"
                stroke="var(--chart-1)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-1)", r: 2 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="tokensPerSec"
                name="Tokens/s"
                stroke="var(--chart-2)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-2)", r: 2 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="batchesPerSec"
                name="Batches/s"
                stroke="var(--chart-3)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-3)", r: 2 }}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function MemoryPanel({
  data,
  onExplain,
}: {
  data: TimeSeriesPoint[];
  onExplain?: () => void;
}) {
  const hasData = data.some(
    (d) =>
      d.processRssMb != null ||
      d.cudaAllocatedMb != null ||
      d.cudaPeakAllocatedMb != null
  );

  return (
    <DashboardPanel category="Resources" title="Memory usage" onExplain={onExplain}>
      {!hasData ? (
        <EmptyState message="No memory data available" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="stepIndex"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <YAxis
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                label={{
                  value: "MB",
                  angle: -90,
                  position: "insideLeft",
                  fill: "var(--muted-foreground)",
                  fontSize: 10,
                }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => (
                  <span className="text-zinc-300">{value}</span>
                )}
              />
              <Area
                type="monotone"
                dataKey="processRssMb"
                name="Process RSS"
                stroke="var(--chart-4)"
                fill="var(--chart-4)"
                fillOpacity={0.15}
                strokeWidth={2}
                connectNulls
              />
              <Area
                type="monotone"
                dataKey="cudaAllocatedMb"
                name="CUDA allocated"
                stroke="var(--chart-1)"
                fill="var(--chart-1)"
                fillOpacity={0.15}
                strokeWidth={2}
                connectNulls
              />
              <Area
                type="monotone"
                dataKey="cudaPeakAllocatedMb"
                name="CUDA peak"
                stroke="var(--chart-5)"
                fill="var(--chart-5)"
                fillOpacity={0.1}
                strokeWidth={1}
                strokeDasharray="4 2"
                connectNulls
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function SystemPanel({
  data,
  onExplain,
}: {
  data: TimeSeriesPoint[];
  onExplain?: () => void;
}) {
  const hasData = data.some(
    (d) => d.cpuPercent != null || d.ramPercent != null
  );

  return (
    <DashboardPanel
      category="Resources"
      title="System utilization"
      onExplain={onExplain}
    >
      {!hasData ? (
        <EmptyState message="No system data available" />
      ) : (
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={data}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="stepIndex"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <YAxis
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                domain={[0, 100]}
                label={{
                  value: "%",
                  angle: -90,
                  position: "insideLeft",
                  fill: "var(--muted-foreground)",
                  fontSize: 10,
                }}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => (
                  <span className="text-zinc-300">{value}</span>
                )}
              />
              <Line
                type="monotone"
                dataKey="cpuPercent"
                name="CPU %"
                stroke="var(--chart-3)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-3)", r: 2 }}
                connectNulls
              />
              <Line
                type="monotone"
                dataKey="ramPercent"
                name="RAM %"
                stroke="var(--chart-2)"
                strokeWidth={2}
                dot={{ fill: "var(--chart-2)", r: 2 }}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function ProfilerOverviewPanel({
  profiler,
  onExplain,
}: {
  profiler: ProfilerSnapshot | null;
  onExplain?: () => void;
}) {
  return (
    <DashboardPanel
      category="Profiler"
      title="Overview"
      onExplain={onExplain}
      badge={profiler ? `${profiler.numUniqueOps} ops` : undefined}
    >
      {!profiler ? (
        <EmptyState message="No profiler data available" />
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <MetricCard
            label="Total CPU time"
            value={`${fmt(profiler.totalCpuMs, 1)} ms`}
          />
          <MetricCard
            label="Total CUDA time"
            value={`${fmt(profiler.totalCudaMs, 1)} ms`}
          />
          <MetricCard
            label="Forward time"
            value={`${fmt(profiler.forwardMs, 1)} ms`}
          />
          <MetricCard
            label="Backward time"
            value={`${fmt(profiler.backwardMs, 1)} ms`}
          />
          <MetricCard
            label="Fwd/Bwd ratio"
            value={fmt(profiler.fwdBwdRatio, 3)}
            sub={
              profiler.fwdBwdRatio < 0.15
                ? "Backward-heavy"
                : profiler.fwdBwdRatio > 0.5
                  ? "Forward-heavy"
                  : "Balanced"
            }
          />
          <MetricCard
            label="Unique ops"
            value={String(profiler.numUniqueOps)}
          />
        </div>
      )}
    </DashboardPanel>
  );
}

const BAR_COLORS = [
  "#10b981", // emerald-500
  "#0ea5e9", // sky-500
  "#f59e0b", // amber-500
  "#8b5cf6", // violet-500
  "#ef4444", // red-500
  "#14b8a6", // teal-500
  "#f97316", // orange-500
  "#ec4899", // pink-500
  "#6366f1", // indigo-500
  "#84cc16", // lime-500
];

function OperationBreakdownPanel({
  profiler,
  onExplain,
}: {
  profiler: ProfilerSnapshot | null;
  onExplain?: () => void;
}) {
  const topOps = profiler?.topOps.slice(0, 10) ?? [];

  return (
    <DashboardPanel
      category="Profiler"
      title="Top operations"
      onExplain={onExplain}
    >
      {topOps.length === 0 ? (
        <EmptyState message="No operation data" />
      ) : (
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={topOps}
              layout="vertical"
              margin={{ top: 4, right: 24, left: 4, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
                horizontal={false}
              />
              <XAxis
                type="number"
                tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                label={{
                  value: "CPU time (ms)",
                  position: "insideBottom",
                  offset: -2,
                  fill: "var(--muted-foreground)",
                  fontSize: 10,
                }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "var(--muted-foreground)", fontSize: 9 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                width={130}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload as TopOp | undefined;
                  if (!d) return null;
                  return (
                    <div className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 shadow-xl">
                      <div className="font-semibold">{d.name}</div>
                      <div>CPU: {fmt(d.cpuTimeMs, 1)} ms</div>
                      <div>Calls: {d.calls}</div>
                      <div>Avg: {fmt(d.avgCpuUs, 1)} us</div>
                    </div>
                  );
                }}
              />
              <Bar dataKey="cpuTimeMs" name="CPU time (ms)" radius={[0, 4, 4, 0]}>
                {topOps.map((_, i) => (
                  <Cell
                    key={i}
                    fill={BAR_COLORS[i % BAR_COLORS.length]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function OpCategoryPanel({
  profiler,
}: {
  profiler: ProfilerSnapshot | null;
}) {
  const cats = profiler?.categories ?? [];

  return (
    <DashboardPanel category="Profiler" title="Op categories">
      {cats.length === 0 ? (
        <EmptyState message="No category data" />
      ) : (
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={cats}
              margin={{ top: 4, right: 24, left: 4, bottom: 4 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="var(--border)"
                opacity={0.5}
              />
              <XAxis
                dataKey="name"
                tick={{ fill: "var(--muted-foreground)", fontSize: 9 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                angle={-30}
                textAnchor="end"
                height={60}
              />
              <YAxis
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                label={{
                  value: "% CPU",
                  angle: -90,
                  position: "insideLeft",
                  fill: "var(--muted-foreground)",
                  fontSize: 10,
                }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0]?.payload as OpCategory | undefined;
                  if (!d) return null;
                  return (
                    <div className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 shadow-xl">
                      <div className="font-semibold">{d.name}</div>
                      <div>{fmt(d.pctCpu, 1)}% CPU</div>
                      <div>{fmt(d.cpuTimeMs, 1)} ms</div>
                    </div>
                  );
                }}
              />
              <Bar
                dataKey="pctCpu"
                name="% CPU"
                radius={[4, 4, 0, 0]}
              >
                {cats.map((_, i) => (
                  <Cell
                    key={i}
                    fill={BAR_COLORS[i % BAR_COLORS.length]}
                    fillOpacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </DashboardPanel>
  );
}

function healthFlagBadge(flag: boolean, label: string, color: "red" | "amber") {
  if (!flag) return null;
  const cls =
    color === "red"
      ? "border-red-500/60 bg-red-950/40 text-red-300"
      : "border-amber-500/60 bg-amber-950/40 text-amber-300";
  return (
    <span
      key={label}
      className={`inline-block rounded-full border px-2 py-0.5 text-[10px] font-medium ${cls}`}
    >
      {label}
    </span>
  );
}

function LayerHealthPanel({
  data,
  onExplain,
}: {
  data: { layers: LayerHealthRow[]; correlations: ActivationCorrelation[] } | null;
  onExplain?: () => void;
}) {
  return (
    <DashboardPanel
      category="Layer health"
      title="Per-layer diagnostics"
      onExplain={onExplain}
      badge={data ? `${data.layers.length} layers` : undefined}
    >
      {!data || data.layers.length === 0 ? (
        <EmptyState message="No layer health data available" />
      ) : (
        <div className="space-y-4">
          <div className="dark-scrollbar max-h-96 overflow-y-auto">
            <table className="w-full text-left text-xs">
              <thead className="sticky top-0 bg-zinc-950">
                <tr className="border-b border-zinc-800 text-[10px] uppercase tracking-wider text-zinc-500">
                  <th className="px-3 py-2 font-medium">Layer</th>
                  <th className="px-3 py-2 font-medium text-right">Act mean</th>
                  <th className="px-3 py-2 font-medium text-right">Act std</th>
                  <th className="px-3 py-2 font-medium text-right">Grad norm</th>
                  <th className="px-3 py-2 font-medium text-right">Wt sparsity</th>
                  <th className="px-3 py-2 font-medium">Flags</th>
                </tr>
              </thead>
              <tbody>
                {data.layers.map((layer) => {
                  const hasIssue =
                    layer.isDead ||
                    layer.hasVanishingGradients ||
                    layer.hasFrozenOutput;
                  const rowBg = hasIssue
                    ? "bg-red-950/10"
                    : "bg-zinc-900/30";

                  return (
                    <tr
                      key={layer.name}
                      className={`border-b border-zinc-800/60 ${rowBg} hover:bg-zinc-800/40`}
                    >
                      <td className="px-3 py-2 font-mono text-zinc-200">
                        {layer.name}
                      </td>
                      <td className="px-3 py-2 font-mono text-right text-zinc-300">
                        {fmt(layer.activationMean, 6)}
                      </td>
                      <td className="px-3 py-2 font-mono text-right text-zinc-300">
                        {fmt(layer.activationStd, 6)}
                      </td>
                      <td className="px-3 py-2 font-mono text-right text-zinc-300">
                        {fmt(layer.gradientNormMean, 6)}
                      </td>
                      <td className="px-3 py-2 font-mono text-right text-zinc-300">
                        {fmt(layer.weightSparsity, 6)}
                      </td>
                      <td className="px-3 py-2">
                        <div className="flex flex-wrap gap-1">
                          {healthFlagBadge(layer.isDead, "Dead", "red")}
                          {healthFlagBadge(
                            layer.hasVanishingGradients,
                            "Vanishing grad",
                            "red"
                          )}
                          {healthFlagBadge(
                            layer.hasFrozenOutput,
                            "Frozen",
                            "red"
                          )}
                          {healthFlagBadge(
                            layer.hasNearZeroWeights,
                            "Near-zero wt",
                            "amber"
                          )}
                          {healthFlagBadge(
                            layer.hasLowActivationVariance,
                            "Low act var",
                            "amber"
                          )}
                          {!layer.isDead &&
                            !layer.hasVanishingGradients &&
                            !layer.hasFrozenOutput &&
                            !layer.hasNearZeroWeights &&
                            !layer.hasLowActivationVariance && (
                              <span className="text-[10px] text-emerald-400/70">
                                Healthy
                              </span>
                            )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {data.correlations.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Activation correlations
              </p>
              <div className="flex flex-wrap gap-2">
                {data.correlations.map((c, i) => {
                  const absCorr = Math.abs(c.correlation);
                  const isHigh = absCorr > 0.95;
                  return (
                    <div
                      key={i}
                      className={`rounded-lg border px-2.5 py-1.5 text-xs ${
                        isHigh
                          ? "border-amber-700/60 bg-amber-950/30 text-amber-300"
                          : "border-zinc-800 bg-zinc-900/40 text-zinc-400"
                      }`}
                    >
                      <span className="font-mono">{c.layerA}</span>
                      <span className="mx-1 text-zinc-600">&harr;</span>
                      <span className="font-mono">{c.layerB}</span>
                      <span className="ml-1.5 font-semibold">
                        {c.correlation.toFixed(3)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </DashboardPanel>
  );
}

function CarbonPanel({
  carbon,
  onExplain,
}: {
  carbon: CarbonSnapshot | null;
  onExplain?: () => void;
}) {
  return (
    <DashboardPanel
      category="Sustainability"
      title="Carbon emissions"
      onExplain={onExplain}
    >
      {!carbon ? (
        <EmptyState message="No carbon emissions data available" />
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <MetricCard
              label="Epoch CO2"
              value={`${fmt(carbon.epochCo2Kg, 6)} kg`}
            />
            <MetricCard
              label="Cumulative CO2"
              value={`${fmt(carbon.cumulativeCo2Kg, 6)} kg`}
            />
            <MetricCard
              label="Power draw"
              value={`${fmt(carbon.powerDrawWatts, 1)} W`}
            />
            <MetricCard
              label="Epoch energy"
              value={`${fmt(carbon.epochEnergyKwh, 6)} kWh`}
            />
          </div>
          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              label="CPU power"
              value={`${fmt(carbon.cpuPowerW, 1)} W`}
              sub={
                carbon.cpuUtilPct != null
                  ? `${fmt(carbon.cpuUtilPct, 1)}% util`
                  : undefined
              }
            />
            <MetricCard
              label="GPU power"
              value={`${fmt(carbon.gpuPowerW, 1)} W`}
              sub={
                carbon.gpuUtilPct != null
                  ? `${fmt(carbon.gpuUtilPct, 1)}% util`
                  : undefined
              }
            />
            <MetricCard
              label="RAM power"
              value={`${fmt(carbon.ramPowerW, 1)} W`}
            />
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-zinc-500">
            {carbon.countryName !== "-" && (
              <span>Region: {carbon.countryName}</span>
            )}
            {carbon.cpuModel !== "-" && <span>CPU: {carbon.cpuModel}</span>}
            {carbon.gpuModel !== "-" && <span>GPU: {carbon.gpuModel}</span>}
          </div>
        </div>
      )}
    </DashboardPanel>
  );
}

// ── Main dashboard component ─────────────────────────────────────────────────

type StepsDashboardProps = {
  steps: TrainStep[];
  stepsLoading: boolean;
  onExplain?: (prompt: string) => void;
};

export default function StepsDashboard({
  steps,
  stepsLoading,
  onExplain,
}: StepsDashboardProps) {
  const [collapsed, setCollapsed] = useState(false);

  const timeSeries = useMemo(() => buildTimeSeries(steps), [steps]);
  const profiler = useMemo(() => extractProfiler(steps), [steps]);
  const layerHealth = useMemo(() => extractLayerHealth(steps), [steps]);
  const carbon = useMemo(() => extractCarbon(steps), [steps]);

  if (stepsLoading) {
    return (
      <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
        <div className="flex h-32 items-center justify-center text-sm text-zinc-500">
          Loading step data...
        </div>
      </section>
    );
  }

  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {/* Dashboard header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold">Step Profiler Dashboard</h2>
          <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
            {steps.length} step{steps.length !== 1 ? "s" : ""}
          </span>
        </div>
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center gap-2 rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-1.5 text-xs text-zinc-400 transition hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
        >
          {collapsed ? "Expand" : "Collapse"}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`transition-transform ${collapsed ? "" : "rotate-180"}`}
            aria-hidden
          >
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </button>
      </div>

      {!collapsed && (
        <>
          {/* Row 1: Loss | Throughput */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <LossTrendPanel
              data={timeSeries}
              onExplain={
                onExplain
                  ? () => onExplain(lossExplainPrompt(timeSeries))
                  : undefined
              }
            />
            <ThroughputPanel
              data={timeSeries}
              onExplain={
                onExplain
                  ? () => onExplain(throughputExplainPrompt(timeSeries))
                  : undefined
              }
            />
          </div>

          {/* Row 2: Memory | System */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <MemoryPanel
              data={timeSeries}
              onExplain={
                onExplain
                  ? () => onExplain(memoryExplainPrompt(timeSeries))
                  : undefined
              }
            />
            <SystemPanel
              data={timeSeries}
              onExplain={
                onExplain
                  ? () => onExplain(systemExplainPrompt(timeSeries))
                  : undefined
              }
            />
          </div>

          {/* Row 3: Profiler overview | Top operations | Op categories */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <ProfilerOverviewPanel
              profiler={profiler}
              onExplain={
                onExplain && profiler
                  ? () => onExplain(profilerExplainPrompt(profiler))
                  : undefined
              }
            />
            <OperationBreakdownPanel
              profiler={profiler}
              onExplain={
                onExplain && profiler
                  ? () => onExplain(profilerExplainPrompt(profiler))
                  : undefined
              }
            />
            <OpCategoryPanel profiler={profiler} />
          </div>

          {/* Row 4: Layer health */}
          <LayerHealthPanel
            data={layerHealth}
            onExplain={
              onExplain && layerHealth
                ? () => onExplain(layerHealthExplainPrompt(layerHealth.layers))
                : undefined
            }
          />

          {/* Row 5: Carbon emissions */}
          <CarbonPanel
            carbon={carbon}
            onExplain={
              onExplain && carbon
                ? () => onExplain(carbonExplainPrompt(carbon))
                : undefined
            }
          />
        </>
      )}
    </div>
  );
}
