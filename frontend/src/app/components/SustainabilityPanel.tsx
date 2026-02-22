"use client";

import type { DiagnosticRunOut, SustainabilityInsight } from "@/lib/client";
import {
  getDiagnosticRunDiagnosticsRunsRunIdGetOptions,
  listSessionDiagnosticRunsDiagnosticsSessionsSessionIdGetOptions,
  runSessionDiagnosticsDiagnosticsSessionsSessionIdRunPostMutation,
} from "@/lib/client/@tanstack/react-query.gen";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo } from "react";

// ── Helpers ──────────────────────────────────────────────────────────────────

function grade(score: number | null | undefined): string {
  if (score == null) return "-";
  if (score >= 0.85) return "A";
  if (score >= 0.7) return "B";
  if (score >= 0.5) return "C";
  if (score >= 0.3) return "D";
  return "F";
}

const GRADE_STYLES: Record<string, string> = {
  A: "bg-emerald-900/40 text-emerald-300 border-emerald-700",
  B: "bg-green-900/40 text-green-300 border-green-700",
  C: "bg-amber-900/40 text-amber-300 border-amber-700",
  D: "bg-orange-900/40 text-orange-300 border-orange-700",
  F: "bg-red-900/40 text-red-300 border-red-700",
  "-": "bg-zinc-800 text-zinc-400 border-zinc-700",
};

function fmt(n: number | null | undefined, digits = 4): string {
  if (n == null) return "-";
  return n.toFixed(digits);
}

function co2Equiv(kg: number): string {
  // 0.21 kg CO₂ per km driven (average EU car)
  const km = kg / 0.21;
  if (km < 1) return `${(km * 1000).toFixed(0)} m driven`;
  return `${km.toFixed(1)} km driven`;
}

// ── Component ────────────────────────────────────────────────────────────────

type SustainabilityPanelProps = {
  sessionId: number | null;
};

export default function SustainabilityPanel({
  sessionId,
}: SustainabilityPanelProps) {
  const queryClient = useQueryClient();

  // 1. Get the list of diagnostic runs for this session
  const { data: runs = [] } = useQuery({
    ...listSessionDiagnosticRunsDiagnosticsSessionsSessionIdGetOptions({
      path: { session_id: sessionId ?? 0 },
    }),
    enabled: sessionId != null,
  });

  // 2. Get the latest run's full detail (which includes sustainability)
  const latestRunId = runs.length > 0 ? runs[0]?.id : null;

  const { data: fullRun, isLoading: isRunLoading } = useQuery({
    ...getDiagnosticRunDiagnosticsRunsRunIdGetOptions({
      path: { run_id: latestRunId ?? 0 },
    }),
    enabled: latestRunId != null,
  });

  const sus: SustainabilityInsight | null = (fullRun as DiagnosticRunOut | undefined)?.sustainability ?? null;

  // 3. Mutation to trigger a fresh diagnostic run
  const runDiagMutation = useMutation({
    ...runSessionDiagnosticsDiagnosticsSessionsSessionIdRunPostMutation(),
    onSuccess: () => {
      queryClient.invalidateQueries();
    },
  });

  const g = grade(sus?.parameter_efficiency_score);
  const gStyle = GRADE_STYLES[g] ?? GRADE_STYLES["-"];

  const totalCo2 = sus?.total_co2_kg;
  const totalEnergy = sus?.total_energy_kwh;
  const wastedCo2 = sus?.wasted_co2_kg;
  const wastedEpochs = sus?.wasted_epochs;
  const wastedPct = sus?.wasted_compute_pct;
  const optimalStop = sus?.optimal_stop_epoch;
  const deadLayers = sus?.dead_layers ?? [];
  const vanishingLayers = sus?.vanishing_gradient_layers ?? [];
  const frozenLayers = sus?.frozen_output_layers ?? [];
  const redundantPairs = sus?.redundant_layer_pairs ?? [];
  const issueCount = sus?.sustainability_issue_count ?? 0;

  // Estimate cost: €50/ton CO₂
  const costEur = totalCo2 != null ? (totalCo2 * 50) / 1000 : null;
  const wastedCostEur = wastedCo2 != null ? (wastedCo2 * 50) / 1000 : null;

  const problemLayers = useMemo(
    () => [
      ...deadLayers.map((l) => ({ layer: l, issue: "Dead neurons" })),
      ...vanishingLayers.map((l) => ({ layer: l, issue: "Vanishing gradients" })),
      ...frozenLayers.map((l) => ({ layer: l, issue: "Frozen output" })),
    ],
    [deadLayers, vanishingLayers, frozenLayers]
  );

  if (sessionId == null) return null;

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Green AI
          </p>
          <h2 className="text-lg font-semibold">Sustainability</h2>
        </div>
        <div className="flex items-center gap-2">
          {/* Efficiency grade badge */}
          <span
            className={`rounded-full border px-3 py-1 text-sm font-bold ${gStyle}`}
            title="Parameter efficiency grade"
          >
            🌱 {g}
          </span>
          <button
            type="button"
            onClick={() =>
              sessionId != null &&
              runDiagMutation.mutate({
                path: { session_id: sessionId },
              })
            }
            disabled={runDiagMutation.isPending || sessionId == null}
            className="rounded-full border border-zinc-700 bg-zinc-800 px-3 py-1 text-xs text-zinc-300 transition hover:border-zinc-500 hover:bg-zinc-700 disabled:opacity-50"
          >
            {runDiagMutation.isPending ? "Running…" : "Refresh"}
          </button>
        </div>
      </div>

      {sus == null && !isRunLoading ? (
        <div className="mt-4 rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/40 px-4 py-5 text-sm text-zinc-400">
          No sustainability data yet. Run diagnostics to generate a report.
        </div>
      ) : isRunLoading ? (
        <div className="mt-4 py-4 text-center text-sm text-zinc-500">
          Loading sustainability data…
        </div>
      ) : (
        <div className="mt-4 space-y-4">
          {/* Carbon footprint row */}
          <div className="grid grid-cols-3 gap-3">
            <MetricCard
              label="Total CO₂"
              value={totalCo2 != null ? `${fmt(totalCo2, 4)} kg` : "-"}
              sub={totalCo2 != null ? `≈ ${co2Equiv(totalCo2)}` : undefined}
              icon="🏭"
            />
            <MetricCard
              label="Total Energy"
              value={totalEnergy != null ? `${fmt(totalEnergy, 4)} kWh` : "-"}
              icon="⚡"
            />
            <MetricCard
              label="Carbon Cost"
              value={costEur != null ? `€${costEur.toFixed(4)}` : "-"}
              sub="@ €50/ton CO₂"
              icon="💰"
            />
          </div>

          {/* Waste analysis */}
          {(wastedEpochs != null && wastedEpochs > 0) || (wastedCo2 != null && wastedCo2 > 0) ? (
            <div className="rounded-2xl border border-amber-800/40 bg-amber-950/20 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-amber-400">
                ⚠ Waste Detected
              </p>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm text-zinc-300">
                {optimalStop != null && (
                  <p>
                    <span className="text-zinc-500">Optimal stop:</span> epoch{" "}
                    {optimalStop}
                  </p>
                )}
                {wastedEpochs != null && (
                  <p>
                    <span className="text-zinc-500">Wasted epochs:</span>{" "}
                    {wastedEpochs}
                  </p>
                )}
                {wastedPct != null && (
                  <p>
                    <span className="text-zinc-500">Wasted compute:</span>{" "}
                    {wastedPct.toFixed(1)}%
                  </p>
                )}
                {wastedCo2 != null && (
                  <p>
                    <span className="text-zinc-500">Wasted CO₂:</span>{" "}
                    {fmt(wastedCo2, 4)} kg
                  </p>
                )}
                {wastedCostEur != null && (
                  <p>
                    <span className="text-zinc-500">Wasted cost:</span>{" "}
                    €{wastedCostEur.toFixed(4)}
                  </p>
                )}
              </div>
            </div>
          ) : null}

          {/* Issue count + power */}
          <div className="flex flex-wrap items-center gap-3 text-xs text-zinc-400">
            <span>
              {issueCount} sustainability issue{issueCount !== 1 ? "s" : ""}
            </span>
            {sus?.avg_power_draw_watts != null && (
              <span>
                Avg power: {sus.avg_power_draw_watts.toFixed(1)} W
              </span>
            )}
            {sus?.total_samples_processed != null && (
              <span>
                {sus.total_samples_processed.toLocaleString()} samples
              </span>
            )}
          </div>

          {/* Problem layers */}
          {problemLayers.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Problem Layers
              </p>
              <div className="space-y-1">
                {problemLayers.slice(0, 8).map((pl, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-1.5 text-xs"
                  >
                    <span className="font-mono text-zinc-300">{pl.layer}</span>
                    <span className="text-red-400">{pl.issue}</span>
                  </div>
                ))}
                {problemLayers.length > 8 && (
                  <p className="pl-1 text-[10px] text-zinc-600">
                    +{problemLayers.length - 8} more
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Redundant layer pairs */}
          {redundantPairs.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                Redundant Layer Pairs
              </p>
              <div className="space-y-1">
                {redundantPairs.slice(0, 4).map((pair, i) => (
                  <div
                    key={i}
                    className="rounded-lg border border-zinc-800 bg-zinc-900/40 px-3 py-1.5 text-xs text-zinc-400"
                  >
                    {JSON.stringify(pair)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// ── Metric card sub-component ────────────────────────────────────────────────

function MetricCard({
  label,
  value,
  sub,
  icon,
}: {
  label: string;
  value: string;
  sub?: string;
  icon: string;
}) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 px-3 py-2.5">
      <p className="text-[10px] uppercase tracking-wider text-zinc-500">
        {icon} {label}
      </p>
      <p className="mt-1 text-sm font-semibold text-zinc-100">{value}</p>
      {sub && <p className="text-[10px] text-zinc-500">{sub}</p>}
    </div>
  );
}
