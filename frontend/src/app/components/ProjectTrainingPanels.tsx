"use client";

import { useState, useEffect, type SVGProps } from "react";
import type { HealthOut, IssueOut, Model, TrainStep } from "@/lib/client";

type TrainSession = {
  id: number;
  projectId: number;
  runId: string;
  runName: string;
  startedAt: string;
  endedAt: string | null;
  device: string;
  cudaAvailable: boolean;
  pytorchVersion: string;
  config: Record<string, unknown> | null;
  summary: Record<string, unknown> | null;
  status: "running" | "completed" | "failed" | "paused" | "pending" | "analyzing";
};

type TrainModel = {
  id: number;
  sessionId: number;
  architecture: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
};

type SessionLog = {
  id: number;
  sessionId: number;
  ts: string;
  level: string;
  msg: string;
  module: string;
  lineno: number;
  kind: "console" | "error";
};

const SAMPLE_SESSIONS: TrainSession[] = [
  {
    id: 17,
    projectId: 1,
    runId: "run_2026_02_21_0715",
    runName: "resnet50-cifar10",
    startedAt: "2026-02-21T07:15:02Z",
    endedAt: null,
    device: "NVIDIA A100",
    cudaAvailable: true,
    pytorchVersion: "2.2.1",
    config: {
      batch_size: 128,
      lr: 0.002,
      optimizer: "AdamW",
      mixed_precision: true,
    },
    summary: {
      best_val_acc: 0.921,
      best_val_loss: 0.312,
      total_epochs: 20,
    },
    status: "running",
  },
  {
    id: 18,
    projectId: 1,
    runId: "run_2026_02_20_2310",
    runName: "resnet34-cifar10",
    startedAt: "2026-02-20T23:10:12Z",
    endedAt: "2026-02-21T00:02:44Z",
    device: "NVIDIA A100",
    cudaAvailable: true,
    pytorchVersion: "2.2.1",
    config: {
      batch_size: 96,
      lr: 0.001,
      optimizer: "AdamW",
      mixed_precision: true,
    },
    summary: {
      best_val_acc: 0.903,
      best_val_loss: 0.354,
      total_epochs: 18,
    },
    status: "completed",
  },
  {
    id: 19,
    projectId: 1,
    runId: "run_2026_02_19_1840",
    runName: "mobilenetv3-cifar10",
    startedAt: "2026-02-19T18:40:55Z",
    endedAt: "2026-02-19T19:22:31Z",
    device: "NVIDIA A10G",
    cudaAvailable: true,
    pytorchVersion: "2.2.1",
    config: {
      batch_size: 256,
      lr: 0.003,
      optimizer: "SGD",
      mixed_precision: false,
    },
    summary: {
      best_val_acc: 0.887,
      best_val_loss: 0.421,
      total_epochs: 15,
    },
    status: "completed",
  },
];

const SAMPLE_MODEL: TrainModel = {
  id: 4,
  sessionId: 17,
  architecture: {
    name: "ResNet50",
    blocks: 50,
    input_shape: "3x32x32",
    num_classes: 10,
  },
  hyperparameters: {
    weight_decay: 0.01,
    dropout: 0.2,
    label_smoothing: 0.1,
  },
};

const SAMPLE_LOGS: SessionLog[] = [
  {
    id: 501,
    sessionId: 17,
    ts: "2026-02-21T08:04:12Z",
    level: "INFO",
    msg: "Epoch 12 finished in 43.2s (val_acc=0.913)",
    module: "trainer",
    lineno: 214,
    kind: "console",
  },
  {
    id: 502,
    sessionId: 17,
    ts: "2026-02-21T08:05:02Z",
    level: "INFO",
    msg: "Epoch 13 finished in 42.7s (val_acc=0.917)",
    module: "trainer",
    lineno: 214,
    kind: "console",
  },
  {
    id: 503,
    sessionId: 17,
    ts: "2026-02-21T08:05:48Z",
    level: "WARN",
    msg: "Gradient overflow detected, scaling down.",
    module: "amp",
    lineno: 88,
    kind: "console",
  },
];

type SessionListProps = {
  selectedProject: { id?: number | null; name?: string } | null;
  sessions: TrainSession[];
  sessionsLoading?: boolean;
  selectedSessionId: number | null;
  onSelectSession: (sessionId: number) => void;
};

function SessionList({
  selectedProject,
  sessions,
  sessionsLoading = false,
  selectedSessionId,
  onSelectSession,
}: SessionListProps) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const badgeText = !mounted
    ? "0 runs"
    : sessionsLoading
      ? "Loading…"
      : `${sessions.length} runs`;

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Train sessions
          </p>
          <h2 className="text-lg font-semibold">Runs per project</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {badgeText}
        </span>
      </div>
      {selectedProject ? (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200">
          {sessions.map((session) => (
            <button
              key={session.id}
              type="button"
              onClick={() => onSelectSession(session.id)}
              className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-left transition ${
                selectedSessionId === session.id
                  ? "border-white/70 bg-white/10 shadow-lg shadow-white/10"
                  : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-600"
              }`}
            >
              <div>
                <p className="text-sm font-semibold text-zinc-100">
                  {session.runName}
                </p>
                <p className="text-xs text-zinc-500">{session.runId}</p>
                <p className="mt-1 text-xs text-zinc-500">
                  {new Date(session.startedAt).toLocaleString()}
                </p>
              </div>
              <div className="flex items-center gap-2 text-xs font-medium">
                <span className="rounded-full bg-zinc-950 px-3 py-1 text-zinc-300">
                  {session.device}
                </span>
                <span className="rounded-full border border-zinc-700 px-3 py-1 text-zinc-300">
                  {session.status}
                </span>
              </div>
            </button>
          ))}
        </div>
      ) : (
        <p className="mt-4 text-xs text-zinc-500">
          Select a project to view associated sessions.
        </p>
      )}
    </section>
  );
}

type TrainSessionPanelProps = {
  session: TrainSession | null;
  onResume?: (sessionId: number) => void;
  onStop?: (sessionId: number) => void;
  actionPending?: boolean;
};

const CONFIG_VISIBLE_INITIAL = 3;

function TrainSessionPanel({
  session,
  onResume,
  onStop,
  actionPending = false,
}: TrainSessionPanelProps) {
  const isPending = session?.status === "pending";
  const [configExpanded, setConfigExpanded] = useState(false);
  const configEntries = session?.config ? Object.entries(session.config) : [];
  const configVisibleCount = configExpanded ? configEntries.length : CONFIG_VISIBLE_INITIAL;
  const configToShow = configEntries.slice(0, configVisibleCount);
  const hasMoreConfig = configEntries.length > CONFIG_VISIBLE_INITIAL;

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Train session
          </p>
          <h2 className="text-lg font-semibold">Run overview</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {session?.status ?? "No session"}
        </span>
      </div>
      {session ? (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200">
          {isPending && onResume && onStop ? (
            <div className="rounded-2xl border border-red-500/50 bg-red-950/40 p-4 shadow-[0_0_20px_-2px_rgba(239,68,68,0.35)]">
              <p className="text-sm text-red-100/95">
                Issues detected. Please check the Session Issues panel to make an
                informed decision on whether to resume or stop the training
                process.
              </p>
              <div className="mt-4 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={() => onResume(session.id)}
                  disabled={actionPending}
                  className="rounded-full border border-emerald-600 bg-emerald-950/50 px-4 py-2 text-xs font-semibold text-emerald-300 transition hover:border-emerald-500 hover:bg-emerald-950/70 disabled:opacity-60"
                >
                  {actionPending ? "…" : "Resume"}
                </button>
                <button
                  type="button"
                  onClick={() => onStop(session.id)}
                  disabled={actionPending}
                  className="rounded-full border border-red-600/60 bg-red-950/30 px-4 py-2 text-xs font-semibold text-red-300 transition hover:border-red-500 hover:bg-red-950/50 disabled:opacity-60"
                >
                  {actionPending ? "…" : "Stop"}
                </button>
              </div>
            </div>
          ) : null}
          <div className="grid gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-semibold text-zinc-100">
                  {session.runName}
                </p>
                <p className="text-xs text-zinc-500">{session.runId}</p>
              </div>
              <span className="rounded-full bg-zinc-950 px-3 py-1 text-xs text-zinc-300">
                {session.device}
              </span>
            </div>
            <div className="flex flex-wrap gap-3 text-xs text-zinc-400">
              <span>
                Started {new Date(session.startedAt).toLocaleString()}
              </span>
              <span>
                Ended{" "}
                {session.endedAt
                  ? new Date(session.endedAt).toLocaleString()
                  : "—"}
              </span>
              <span>
                CUDA {session.cudaAvailable ? "available" : "not detected"}
              </span>
              <span>PyTorch {session.pytorchVersion}</span>
            </div>
          </div>
          <div className="grid gap-3 md:grid-cols-1">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                Config
              </p>
              <div className="mt-3 grid gap-2">
                {session.config ? (
                  <>
                    {configToShow.map(([key, value]) => (
                      <div
                        key={key}
                        className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300"
                      >
                        <span className="uppercase tracking-[0.2em] text-zinc-500">
                          {key.replace(/_/g, " ")}
                        </span>
                        <span className="font-medium text-zinc-200">
                          {String(value)}
                        </span>
                      </div>
                    ))}
                    {hasMoreConfig ? (
                      <button
                        type="button"
                        onClick={() => setConfigExpanded((e) => !e)}
                        className="mt-1 rounded-lg border border-zinc-700 bg-zinc-800/60 px-3 py-1.5 text-xs text-zinc-400 transition hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-300"
                      >
                        {configExpanded
                          ? "Show less"
                          : `Show ${configEntries.length - CONFIG_VISIBLE_INITIAL} more`}
                      </button>
                    ) : null}
                  </>
                ) : (
                  <p className="text-xs text-zinc-500">
                    No config captured.
                  </p>
                )}
              </div>
            </div>
            {/*
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                Summary
              </p>
              <div className="mt-3 grid gap-2">
                {session.summary ? (
                  Object.entries(session.summary).map(([key, value]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300"
                    >
                      <span className="uppercase tracking-[0.2em] text-zinc-500">
                        {key.replace(/_/g, " ")}
                      </span>
                      <span className="font-medium text-zinc-200">
                        {String(value)}
                      </span>
                    </div>
                  ))
                ) : (
                  <p className="text-xs text-zinc-500">
                    No summary metrics yet.
                  </p>
                )
                TODO: Add summary metrics
                  }
              </div>
            </div>
            */}
          </div>
        </div>
      ) : (
        <p className="mt-4 text-xs text-zinc-500">
          Select a project to view the latest training run.
        </p>
      )}
    </section>
  );
}

type ModelPanelProps = {
  session: TrainSession | null;
  model: Model | null | undefined;
};

function ModelPanel({ session, model }: ModelPanelProps) {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Model
          </p>
          <h2 className="text-lg font-semibold">Architecture & params</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          Session {session ? `#${session.id}` : "—"}
        </span>
      </div>
      {!session ? (
        <p className="mt-4 text-xs text-zinc-500">
          Select a run to inspect the bound model definition.
        </p>
      ) : model === undefined ? (
        <p className="mt-4 text-xs text-zinc-500">Loading model…</p>
      ) : model === null ? (
        <p className="mt-4 text-xs text-zinc-500">
          No model registered for this session.
        </p>
      ) : (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200 md:grid-cols-2">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
              Architecture
            </p>
            <div className="mt-3 grid gap-2">
              {Object.entries(model.architecture ?? {})
                .filter(
                  ([key]) =>
                    !["layers", "module_tree", "layer_graph"].includes(key)
                )
                .map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300"
                >
                  <span className="uppercase tracking-[0.2em] text-zinc-500">
                    {key.replace(/_/g, " ")}
                  </span>
                  <span className="font-medium text-zinc-200">
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
              Hyperparameters
            </p>
            <div className="mt-3 grid gap-2">
              {Object.entries(model.hyperparameters ?? {}).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2 text-xs text-zinc-300"
                >
                  <span className="uppercase tracking-[0.2em] text-zinc-500">
                    {key.replace(/_/g, " ")}
                  </span>
                  <span className="font-medium text-zinc-200">
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

type TrainStepListProps = {
  session: TrainSession | null;
  steps: TrainStep[];
  stepsLoading?: boolean;
};

function TrainStepList({ session, steps, stepsLoading = false }: TrainStepListProps) {
  const [expandedStepKey, setExpandedStepKey] = useState<number | null>(null);

  const toggleStep = (key: number) => {
    setExpandedStepKey((prev) => (prev === key ? null : key));
  };

  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Train steps
          </p>
          <h2 className="text-lg font-semibold">Recent steps</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {stepsLoading ? "Loading…" : `${steps.length} steps`}
        </span>
      </div>
      {!session ? (
        <p className="mt-4 text-xs text-zinc-500">
          Select a run to review recent training steps.
        </p>
      ) : stepsLoading ? (
        <p className="mt-4 text-xs text-zinc-500">Loading steps…</p>
      ) : (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200">
          {steps.map((step) => {
            const stepKey = step.id ?? step.step_index;
            const isExpanded = expandedStepKey === stepKey;

            const sections = [
              { title: "Loss", data: step.loss },
              { title: "Throughput", data: step.throughput },
              { title: "Profiler", data: step.profiler },
              { title: "Memory", data: step.memory },
              { title: "System", data: step.system },
              { title: "Layer health", data: step.layer_health ?? undefined },
              { title: "Sustainability", data: step.sustainability ?? undefined },
              { title: "Carbon emissions", data: step.carbon_emissions ?? undefined },
              { title: "Log counts", data: step.log_counts ?? undefined },
            ].filter((s): s is { title: string; data: Record<string, unknown> } => s.data != null && Object.keys(s.data).length > 0);

            return (
              <div
                key={stepKey}
                className="rounded-2xl border border-zinc-800 bg-zinc-900/40 overflow-hidden"
              >
                <button
                  type="button"
                  onClick={() => toggleStep(stepKey)}
                  aria-expanded={isExpanded}
                  className="flex w-full flex-wrap items-center justify-between gap-2 p-4 text-left hover:bg-zinc-800/40"
                >
                  <p className="text-sm font-semibold text-zinc-100">
                    Step {step.step_index}
                  </p>
                  <span className="flex items-center gap-2 text-xs text-zinc-500">
                    {step.duration_seconds.toFixed(1)}s
                    <span
                      className={`inline-block transition-transform ${isExpanded ? "rotate-180" : ""}`}
                      aria-hidden
                    >
                      ▼
                    </span>
                  </span>
                </button>
                <div className="border-t border-zinc-800 px-4 pb-4 pt-2 text-xs text-zinc-400">
                  {new Date(step.timestamp).toLocaleString()}
                </div>
                {isExpanded && (
                  <div className="flex flex-col gap-3 border-t border-zinc-800 p-4 pt-3">
                    {sections.map(({ title, data }) => {
                      const scalarEntries = Object.entries(data).filter(
                        ([, value]) =>
                          value === null ||
                          typeof value === "string" ||
                          typeof value === "number" ||
                          typeof value === "boolean"
                      );
                      if (scalarEntries.length === 0) return null;
                      return (
                        <div
                          key={title}
                          className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 p-3"
                        >
                          <p className="mb-2 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                            {title}
                          </p>
                          <div className="flex flex-col gap-1.5">
                            {scalarEntries.map(([key, value]) => (
                              <div
                                key={key}
                                className="flex items-center justify-between gap-2 text-xs text-zinc-300"
                              >
                                <span className="truncate uppercase tracking-widest text-zinc-500">
                                  {key.replace(/_/g, " ")}
                                </span>
                                <span className="shrink-0 font-medium text-zinc-200">
                                  {value === null ? "—" : String(value)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

type SessionLogListProps = {
  session: TrainSession | null;
  logs: SessionLog[];
  logsLoading?: boolean;
};

function SessionLogList({ session, logs, logsLoading }: SessionLogListProps) {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Session logs
          </p>
          <h2 className="text-lg font-semibold">Runtime output</h2>
        </div>
      </div>
      {session ? (
        <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950/80">
          <div className="flex items-center justify-between gap-2 border-b border-zinc-800 px-4 py-2 text-xs text-zinc-500">
            <span className="uppercase tracking-[0.2em]">Console</span>
            <span>{logsLoading ? "Loading…" : `${logs.length} lines`}</span>
          </div>
          <div className="px-4 py-3 font-mono text-xs text-zinc-200">
            {logsLoading ? (
              <p className="py-4 text-center text-zinc-500">Loading logs…</p>
            ) : (
              logs.map((log) => (
                <div key={log.id} className="flex flex-wrap gap-2 py-1">
                  <span className="text-zinc-500">
                    {new Date(log.ts).toLocaleTimeString()}
                  </span>
                  <span
                    className={`font-semibold ${
                      log.level === "ERROR" || log.kind === "error"
                        ? "text-red-300"
                        : log.level === "WARN"
                          ? "text-amber-300"
                          : "text-emerald-300"
                    }`}
                  >
                    {log.level}
                  </span>
                  <span className="text-zinc-500">{log.module}</span>
                  <span className="text-zinc-500">:{log.lineno}</span>
                  <span className="text-zinc-200">{log.msg}</span>
                </div>
              ))
            )}
          </div>
        </div>
      ) : (
        <p className="mt-4 text-xs text-zinc-500">
          Select a project to inspect live session logs.
        </p>
      )}
    </section>
  );
}

type SessionIssuesPanelProps = {
  session: TrainSession | null;
  health: HealthOut | null;
  healthLoading: boolean;
};

function severityBadgeClass(severity: IssueOut["severity"]): string {
  switch (severity) {
    case "critical":
      return "border-red-500/60 bg-red-950/40 text-red-300";
    case "warning":
      return "border-amber-500/60 bg-amber-950/40 text-amber-300";
    case "info":
    default:
      return "border-sky-500/60 bg-sky-950/40 text-sky-300";
  }
}

function SuggestionBlock({ suggestion }: { suggestion: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(suggestion);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="mt-3 rounded-xl border border-emerald-800/60 bg-emerald-950/30 px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wider text-emerald-400/90">
          Suggested fix
        </p>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/60 px-2.5 py-1.5 text-xs text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-100"
          title="Copy suggestion"
        >
          {copied ? (
            "Copied!"
          ) : (
            <>
              <CopyIcon className="h-3.5 w-3.5" aria-hidden />
              Copy
            </>
          )}
        </button>
      </div>
      <p className="mt-1 text-sm text-zinc-200">{suggestion}</p>
    </div>
  );
}

function CopyIcon({ className, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      {...props}
    >
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16V4a2 2 0 0 1 2-2h10" />
    </svg>
  );
}

function SessionIssuesPanel({
  session,
  health,
  healthLoading,
}: SessionIssuesPanelProps) {
  const issueCount = health?.issues?.length ?? 0;
  const badgeLabel =
    health != null
      ? `Score: ${health.health_score} · ${issueCount} issue${issueCount !== 1 ? "s" : ""}`
      : "—";
  const showGlow =
    session?.status === "pending" && issueCount > 0;

  return (
    <section
      className={`rounded-3xl border bg-zinc-950/60 p-6 shadow-lg ${
        showGlow
          ? "border-red-500/50 shadow-[0_0_24px_-2px_rgba(239,68,68,0.4)]"
          : "border-zinc-800"
      }`}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Run health
          </p>
          <h2 className="text-lg font-semibold">Session issues</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {badgeLabel}
        </span>
      </div>
      {!session ? (
        <p className="mt-4 text-xs text-zinc-500">
          Select a run to view issues and suggested fixes.
        </p>
      ) : healthLoading ? (
        <p className="mt-4 text-xs text-zinc-500">Loading…</p>
      ) : health == null ? (
        <p className="mt-4 text-xs text-zinc-500">
          No health data available for this session.
        </p>
      ) : issueCount === 0 ? (
        <p className="mt-4 text-sm text-zinc-400">
          No issues reported. Health score: {health.health_score}
        </p>
      ) : (
        <div className="dark-scrollbar mt-4 grid max-h-96 gap-3 overflow-y-auto pr-2">
          {health.issues.map((issue, idx) => (
            <div
              key={issue.id ?? idx}
              className="rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4 text-sm"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`rounded-full border px-2.5 py-0.5 text-xs font-medium uppercase tracking-wider ${severityBadgeClass(issue.severity)}`}
                >
                  {issue.severity}
                </span>
                <span className="rounded-full bg-zinc-800 px-2.5 py-0.5 text-xs text-zinc-400">
                  {issue.category}
                </span>
                {issue.epoch_index != null ? (
                  <span className="text-xs text-zinc-500">
                    Epoch {issue.epoch_index}
                  </span>
                ) : null}
                {issue.layer_id ? (
                  <span className="font-mono text-xs text-zinc-500">
                    {issue.layer_id}
                  </span>
                ) : null}
              </div>
              <p className="mt-2 font-semibold text-zinc-100">{issue.title}</p>
              {issue.description ? (
                <p className="mt-1 text-xs text-zinc-400">{issue.description}</p>
              ) : null}
              {issue.suggestion ? (
                <SuggestionBlock
                  suggestion={issue.suggestion}
                />
              ) : null}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export {
  SAMPLE_LOGS,
  SAMPLE_MODEL,
  SAMPLE_SESSIONS,
  ModelPanel,
  SessionIssuesPanel,
  SessionList,
  SessionLogList,
  TrainSessionPanel,
  TrainStepList,
};

export type { SessionLog, TrainSession, TrainStep };
