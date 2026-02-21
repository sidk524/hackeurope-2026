"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getProjectTrendDiagnosticsProjectsProjectIdTrendGetOptions } from "@/lib/client/@tanstack/react-query.gen";
import type { SessionTrendItem } from "@/lib/client";

type ProjectTrendChartProps = {
  projectId: number | null;
};

type ChartPoint = {
  index: number;
  name: string;
  run_id: string;
  started_at: string;
  diagnostic_run_count: number;
  final_train_loss: number | null;
  final_val_loss: number | null;
  val_acc_pct: number | null;
  health_score: number | null;
};

function formatShortDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso.slice(0, 16);
  }
}

function buildChartData(sessions: SessionTrendItem[]): ChartPoint[] {
  return sessions.map((s, i) => ({
    index: i + 1,
    name: `${i + 1} (${s.diagnostic_run_count ?? 0})`,
    run_id: s.run_id,
    started_at: s.started_at,
    diagnostic_run_count: s.diagnostic_run_count ?? 0,
    final_train_loss: s.final_train_loss ?? null,
    final_val_loss: s.final_val_loss ?? null,
    val_acc_pct: s.val_acc != null ? s.val_acc * 100 : null,
    health_score: s.health_score ?? null,
  }));
}

function hasAnyMetric(p: ChartPoint): boolean {
  return (
    p.final_train_loss != null ||
    p.final_val_loss != null ||
    p.val_acc_pct != null ||
    p.health_score != null
  );
}

function CustomTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ name: string; value?: number; dataKey: string; color?: string; payload?: ChartPoint }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  const lines: string[] = [
    `Session ${point.index} · ${point.diagnostic_run_count} diagnostic run${point.diagnostic_run_count !== 1 ? "s" : ""}`,
    point.run_id,
    `Started: ${formatShortDate(point.started_at)}`,
  ];
  if (point.final_train_loss != null) lines.push(`Train loss: ${point.final_train_loss.toFixed(4)}`);
  if (point.final_val_loss != null) lines.push(`Val loss: ${point.final_val_loss.toFixed(4)}`);
  if (point.val_acc_pct != null) lines.push(`Val acc: ${point.val_acc_pct.toFixed(1)}%`);
  if (point.health_score != null) lines.push(`Health: ${point.health_score}`);
  return (
    <div className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs text-zinc-200 shadow-xl">
      {lines.map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );
}

export default function ProjectTrendChart({ projectId }: ProjectTrendChartProps) {
  const {
    data: trend,
    isLoading,
    isError,
    error,
  } = useQuery({
    ...getProjectTrendDiagnosticsProjectsProjectIdTrendGetOptions({
      path: { project_id: projectId ?? 0 },
    }),
    enabled: projectId != null,
  });

  const { chartData, hasTrainLoss, hasValLoss, hasValAcc, hasHealth } = useMemo(() => {
    if (!trend?.sessions?.length) {
      return {
        chartData: [] as ChartPoint[],
        hasTrainLoss: false,
        hasValLoss: false,
        hasValAcc: false,
        hasHealth: false,
      };
    }
    const data = buildChartData(trend.sessions);
    const withMetric = data.filter(hasAnyMetric);
    return {
      chartData: data,
      hasTrainLoss: withMetric.some((p) => p.final_train_loss != null),
      hasValLoss: withMetric.some((p) => p.final_val_loss != null),
      hasValAcc: withMetric.some((p) => p.val_acc_pct != null),
      hasHealth: withMetric.some((p) => p.health_score != null),
    };
  }, [trend?.sessions]);

  const hasAnySeries = hasTrainLoss || hasValLoss || hasValAcc || hasHealth;
  const sessions = trend?.sessions ?? [];
  const improving = trend?.improving;

  if (projectId == null) {
    return (
      <section
        className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg"
        aria-label="Project trend"
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Diagnostics</p>
            <h2 className="text-lg font-semibold">Project trend</h2>
            <p className="mt-1 text-sm text-zinc-500">
              Metrics per training session. X-axis: session index (number of diagnostic runs).
            </p>
          </div>
        </div>
        <p className="mt-4 text-sm text-zinc-500">Select a project to view improvement trend.</p>
      </section>
    );
  }

  if (isLoading) {
    return (
      <section
        className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg"
        aria-label="Project trend"
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Diagnostics</p>
            <h2 className="text-lg font-semibold">Project trend</h2>
            <p className="mt-1 text-sm text-zinc-500">
              Metrics per training session. X-axis: session index (number of diagnostic runs).
            </p>
          </div>
        </div>
        <div className="mt-4 flex h-64 items-center justify-center text-sm text-zinc-500">
          Loading trend…
        </div>
      </section>
    );
  }

  if (isError) {
    return (
      <section
        className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg"
        aria-label="Project trend"
      >
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Diagnostics</p>
            <h2 className="text-lg font-semibold">Project trend</h2>
            <p className="mt-1 text-sm text-zinc-500">
              Metrics per training session. X-axis: session index (number of diagnostic runs).
            </p>
          </div>
        </div>
        <div className="mt-4 rounded-2xl border border-red-900/50 bg-red-950/20 px-4 py-3 text-sm text-red-300">
          Failed to load trend: {error instanceof Error ? error.message : "Unknown error"}
        </div>
      </section>
    );
  }

  return (
    <section
      className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg"
      aria-label="Project trend"
    >
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">Diagnostics</p>
          <h2 className="text-lg font-semibold">Project trend</h2>
          <p className="mt-1 text-sm text-zinc-500">
            Metrics per training session. X-axis: session index (number of diagnostic runs).
          </p>
        </div>
        {improving !== undefined && improving !== null && (
          <span
            className={`rounded-full px-3 py-1 text-xs font-medium ${
              improving
                ? "border border-emerald-700/60 bg-emerald-950/50 text-emerald-300"
                : "border border-amber-800/60 bg-amber-950/40 text-amber-300"
            }`}
          >
            Trend: {improving ? "Improving" : "Not improving"}
          </span>
        )}
        {improving === null && sessions.length >= 2 && (
          <span className="rounded-full border border-zinc-700 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
            Trend: Inconclusive
          </span>
        )}
      </div>

      {!sessions.length ? (
        <p className="mt-4 text-sm text-zinc-500">
          No session data yet. Run training and diagnostics to see the trend.
        </p>
      ) : !hasAnySeries ? (
        <p className="mt-4 text-sm text-zinc-500">
          No metrics available for these sessions yet. Complete runs with loss/diagnostics to see the trend.
        </p>
      ) : (
        <div className="mt-4 h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={chartData}
              margin={{ top: 8, right: 24, left: 8, bottom: 8 }}
              accessibilityLayer
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
              <XAxis
                dataKey="name"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
              />
              <YAxis
                yAxisId="loss"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                domain={["auto", "auto"]}
                hide={!hasTrainLoss && !hasValLoss}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
                tickLine={{ stroke: "var(--border)" }}
                axisLine={{ stroke: "var(--border)" }}
                domain={[0, 100]}
                hide={!hasValAcc && !hasHealth}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ fontSize: 11 }}
                formatter={(value) => <span className="text-zinc-300">{value}</span>}
              />
              {hasTrainLoss && (
                <Line
                  type="monotone"
                  dataKey="final_train_loss"
                  name="Train loss"
                  yAxisId="loss"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-1)", r: 3 }}
                  connectNulls
                />
              )}
              {hasValLoss && (
                <Line
                  type="monotone"
                  dataKey="final_val_loss"
                  name="Val loss"
                  yAxisId="loss"
                  stroke="var(--chart-2)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-2)", r: 3 }}
                  connectNulls
                />
              )}
              {hasValAcc && (
                <Line
                  type="monotone"
                  dataKey="val_acc_pct"
                  name="Val acc %"
                  yAxisId="right"
                  stroke="var(--chart-3)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-3)", r: 3 }}
                  connectNulls
                />
              )}
              {hasHealth && (
                <Line
                  type="monotone"
                  dataKey="health_score"
                  name="Health score"
                  yAxisId="right"
                  stroke="var(--chart-4)"
                  strokeWidth={2}
                  dot={{ fill: "var(--chart-4)", r: 3 }}
                  connectNulls
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
