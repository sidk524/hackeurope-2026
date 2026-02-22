"use client";

import type { HealthOut, IssueOut, Model, TrainStep } from "@/lib/client";
import {
  useAgentChat,
  type AgentMessage,
  type BeliefState,
} from "@/lib/use-agent-chat";
import { useCallback, useEffect, useRef, useState, type SVGProps } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

function SuggestionBlock({ suggestion, issueId }: { suggestion: string; issueId?: number | null }) {
  const [copied, setCopied] = useState(false);
  const [promptCopied, setPromptCopied] = useState(false);
  const [prompt, setPrompt] = useState<string | null>(null);
  const [promptLoading, setPromptLoading] = useState(false);
  const [promptError, setPromptError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(suggestion);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  };

  const handleCreatePrompt = async () => {
    if (!issueId) return;
    if (prompt) {
      // Already generated — toggle visibility
      setExpanded((prev) => !prev);
      return;
    }
    setPromptLoading(true);
    setPromptError(null);
    try {
      const res = await fetch(`${API_BASE}/diagnostics/issues/${issueId}/prompt`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setPrompt(data.prompt);
      setExpanded(true);
    } catch (err: unknown) {
      setPromptError(err instanceof Error ? err.message : "Failed to generate prompt");
    } finally {
      setPromptLoading(false);
    }
  };

  const handleCopyPrompt = async () => {
    if (!prompt) return;
    try {
      await navigator.clipboard.writeText(prompt);
      setPromptCopied(true);
      setTimeout(() => setPromptCopied(false), 2000);
    } catch {
      setPromptCopied(false);
    }
  };

  return (
    <div className="mt-3 rounded-xl border border-emerald-800/60 bg-emerald-950/30 px-3 py-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-medium uppercase tracking-wider text-emerald-400/90">
          Suggested fix
        </p>
        <div className="flex items-center gap-2">
          {issueId ? (
            <button
              type="button"
              onClick={handleCreatePrompt}
              disabled={promptLoading}
              className="flex items-center gap-1.5 rounded-lg border border-emerald-700 bg-emerald-950/60 px-2.5 py-1.5 text-xs text-emerald-300 transition hover:border-emerald-500 hover:bg-emerald-900 hover:text-emerald-100 disabled:opacity-50"
              title="Generate an LLM prompt to solve this issue"
            >
              {promptLoading ? (
                "Generating…"
              ) : prompt ? (
                <>
                  <PromptIcon className="h-3.5 w-3.5" aria-hidden />
                  {expanded ? "Hide Prompt" : "Show Prompt"}
                </>
              ) : (
                <>
                  <PromptIcon className="h-3.5 w-3.5" aria-hidden />
                  Create Prompt
                </>
              )}
            </button>
          ) : null}
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
      </div>
      <p className="mt-1 text-sm text-zinc-200">{suggestion}</p>

      {promptError ? (
        <p className="mt-2 text-xs text-red-400">{promptError}</p>
      ) : null}

      {prompt && expanded ? (
        <div className="mt-3 rounded-lg border border-violet-800/60 bg-violet-950/30 px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-medium uppercase tracking-wider text-violet-400/90">
              Generated Prompt
            </p>
            <button
              type="button"
              onClick={handleCopyPrompt}
              className="flex items-center gap-1.5 rounded-lg border border-zinc-700 bg-zinc-900/60 px-2.5 py-1.5 text-xs text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-100"
              title="Copy prompt"
            >
              {promptCopied ? (
                "Copied!"
              ) : (
                <>
                  <CopyIcon className="h-3.5 w-3.5" aria-hidden />
                  Copy Prompt
                </>
              )}
            </button>
          </div>
          <pre className="dark-scrollbar mt-2 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-zinc-200 font-mono leading-relaxed">
            {prompt}
          </pre>
        </div>
      ) : null}
    </div>
  );
}

function PromptIcon({ className, ...props }: SVGProps<SVGSVGElement>) {
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
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
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
      className={`flex min-h-0 max-h-80 flex-col overflow-hidden rounded-3xl border bg-zinc-950/60 p-6 shadow-lg ${
        showGlow
          ? "border-red-500/50 shadow-[0_0_24px_-2px_rgba(239,68,68,0.4)]"
          : "border-zinc-800"
      }`}
    >
      <div className="shrink-0 flex flex-wrap items-center justify-between gap-3">
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
      <div className="dark-scrollbar min-h-0 flex-1 overflow-y-auto pr-2">
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
        <div className="mt-4 grid gap-3">
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
                  issueId={issue.id}
                />
              ) : null}
            </div>
          ))}
        </div>
      )}
      </div>
    </section>
  );
}

// ── Agent Terminal helpers (inline) ──────────────────────────────────────────

const SEVERITY_COLORS: Record<string, string> = {
  healthy: "text-emerald-400",
  watch: "text-blue-400",
  warning: "text-amber-400",
  critical: "text-red-400",
};

const GRADE_COLORS: Record<string, string> = {
  A: "text-emerald-400",
  B: "text-green-300",
  C: "text-amber-300",
  D: "text-orange-400",
  F: "text-red-400",
};

const QUICK_ACTIONS = [
  { label: "health", prompt: "Give me a health summary of this training run. What are the top issues?" },
  { label: "sustainability", prompt: "Generate a full sustainability report. How green is this training run?" },
  { label: "architecture", prompt: "Analyze the model architecture. Are there any inefficiencies or suggestions?" },
  { label: "diagnose", prompt: "Run a fresh diagnostic analysis and tell me what you find." },
] as const;

function BeliefStatusBar({ belief }: { belief: BeliefState }) {
  const [expanded, setExpanded] = useState(false);
  const sev = belief.severity ?? "watch";
  const grade = belief.sustainability_grade ?? "-";

  return (
    <div className="border-b border-zinc-800 bg-zinc-950/80 px-4 py-2">
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        className="flex w-full items-center gap-3 text-left font-mono text-xs"
      >
        <span className="text-zinc-600">belief</span>
        <span className={`font-semibold uppercase ${SEVERITY_COLORS[sev] ?? "text-zinc-400"}`}>
          {sev}
        </span>
        <span className="text-zinc-600">|</span>
        <span className={`font-bold ${GRADE_COLORS[grade] ?? "text-zinc-400"}`}>{grade}</span>
        <span className="text-zinc-600">|</span>
        <span className="text-zinc-500">confidence {Math.round(belief.confidence * 100)}%</span>
        {belief.revision_count > 0 && (
          <>
            <span className="text-zinc-600">|</span>
            <span className="text-amber-400/80">{belief.revision_count} rev</span>
          </>
        )}
        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={`ml-auto text-zinc-600 transition ${expanded ? "rotate-180" : ""}`}><polyline points="6 9 12 15 18 9" /></svg>
      </button>
      {expanded && (
        <div className="mt-2 space-y-1.5 border-t border-zinc-800/60 pt-2 font-mono text-[11px]">
          <p className="text-zinc-400"><span className="text-zinc-600">issue:</span> {belief.primary_issue}</p>
          {belief.key_observations?.length > 0 && (
            <div className="text-zinc-400">
              <span className="text-zinc-600">observations:</span>
              {belief.key_observations.map((o, i) => <p key={i} className="ml-3 text-zinc-500">- {o}</p>)}
            </div>
          )}
          {belief.recommended_actions?.length > 0 && (
            <div className="text-zinc-400">
              <span className="text-zinc-600">actions:</span>
              {belief.recommended_actions.map((a, i) => <p key={i} className="ml-3 text-emerald-400/70">→ {a}</p>)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TerminalEntry({ msg }: { msg: AgentMessage }) {
  const isUser = msg.role === "user";
  const displayContent = msg.content
    .replace(/<belief>[\s\S]*?```json[\s\S]*?```/g, "")
    .replace(/<belief>[\s\S]*?\{[\s\S]*?\}/g, "")
    .trim();

  if (isUser) {
    return (
      <div className="group flex items-start gap-2 py-1">
        <span className="shrink-0 select-none font-mono text-xs font-bold text-indigo-400">$</span>
        <span className="font-mono text-xs text-zinc-100">{msg.content}</span>
      </div>
    );
  }

  return (
    <div className="py-1.5 pl-4 border-l-2 border-zinc-800/60">
      <div className="mb-1 flex items-center gap-2">
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-[8px] font-bold text-white">A</span>
        <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-indigo-400/80">atlas</span>
        {msg.model && <span className="font-mono text-[9px] text-zinc-700">{msg.model}</span>}
      </div>
      <div className="prose prose-invert prose-sm max-w-none break-words font-mono text-xs leading-relaxed prose-p:my-1 prose-headings:my-1.5 prose-headings:text-zinc-200 prose-headings:text-xs prose-li:my-0 prose-ul:my-0.5 prose-ol:my-0.5 prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-indigo-300 prose-code:text-[11px] prose-code:before:content-none prose-code:after:content-none prose-pre:my-1.5 prose-pre:rounded-lg prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800 prose-a:text-indigo-400 prose-strong:text-zinc-100 prose-blockquote:border-indigo-800/50 prose-blockquote:text-zinc-500 prose-table:text-[10px] prose-th:text-zinc-300 prose-td:text-zinc-400">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayContent}</ReactMarkdown>
      </div>
    </div>
  );
}

// ── Bottom Terminal Panel (Runtime output + Agent) ───────────────────────────

type BottomTerminalPanelProps = {
  /** Active training session (drives logs + status indicator) */
  session: TrainSession | null;
  /** Logs for the session */
  logs: SessionLog[];
  logsLoading?: boolean;
  /** Console behaviour */
  isOpen: boolean;
  onToggleOpen: () => void;
  consoleHeight: number;
  onDragStart: (e: React.MouseEvent) => void;
  consoleFollow: boolean;
  onToggleFollow: () => void;
  consoleBodyRef: React.RefObject<HTMLDivElement | null>;
  onConsoleScroll: (e: React.UIEvent<HTMLDivElement>) => void;
  /** Agent */
  sessionId: number | null;
  projectId: number | null;
};

function BottomTerminalPanel({
  session,
  logs,
  logsLoading = false,
  isOpen,
  onToggleOpen,
  consoleHeight,
  onDragStart,
  consoleFollow,
  onToggleFollow,
  consoleBodyRef,
  onConsoleScroll,
  sessionId,
  projectId,
}: BottomTerminalPanelProps) {
  // Tab state: "logs" (left) or "agent" (right)
  const [activeTab, setActiveTab] = useState<"logs" | "agent">("logs");

  // ── Agent state ──
  const [input, setInput] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const agentScrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { messages, beliefState, isLoading: isAgentLoading, error: agentError, sendMessage, clearHistory } =
    useAgentChat({ sessionId, projectId });

  // Auto-scroll agent output
  useEffect(() => {
    if (agentScrollRef.current) {
      agentScrollRef.current.scrollTop = agentScrollRef.current.scrollHeight;
    }
  }, [messages, isAgentLoading]);

  const focusInput = useCallback(() => { inputRef.current?.focus(); }, []);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = input.trim();
      if (!trimmed) return;
      if (trimmed === "clear") {
        clearHistory();
        setInput("");
        setHistoryIndex(-1);
        return;
      }
      setCommandHistory((prev) => [trimmed, ...prev].slice(0, 50));
      setHistoryIndex(-1);
      sendMessage(trimmed);
      setInput("");
    },
    [input, sendMessage, clearHistory]
  );

  const handleQuickAction = useCallback(
    (prompt: string) => { sendMessage(prompt); focusInput(); },
    [sendMessage, focusInput]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
      if (e.key === "ArrowUp") {
        e.preventDefault();
        setHistoryIndex((prev) => {
          const next = Math.min(prev + 1, commandHistory.length - 1);
          if (next >= 0 && commandHistory[next]) setInput(commandHistory[next]);
          return next;
        });
      }
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setHistoryIndex((prev) => {
          const next = prev - 1;
          if (next < 0) { setInput(""); return -1; }
          if (commandHistory[next]) setInput(commandHistory[next]);
          return next;
        });
      }
    },
    [handleSubmit, commandHistory]
  );

  const noSession = sessionId == null;
  const hasErrors = logs.some((l) => l.kind === "error" || l.level === "ERROR");

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex flex-col shadow-[0_-4px_24px_rgba(0,0,0,0.4)]">
      {/* Drag handle */}
      <div
        onMouseDown={onDragStart}
        className="group flex h-2 cursor-ns-resize items-center justify-center bg-zinc-900 hover:bg-zinc-800"
        aria-hidden
      >
        <div className="h-0.5 w-8 rounded-full bg-zinc-700 group-hover:bg-zinc-500 transition-colors" />
      </div>

      {/* Header bar with tabs */}
      <div className="flex items-center justify-between border-t border-zinc-700 bg-zinc-900 px-5 py-0 select-none">
        {/* Left: tabs */}
        <div className="flex items-center gap-0">
          {/* Runtime output tab */}
          <button
            type="button"
            onClick={() => { setActiveTab("logs"); if (!isOpen) onToggleOpen(); }}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 font-mono text-xs font-semibold uppercase tracking-widest transition ${
              activeTab === "logs" && isOpen
                ? "border-emerald-500 text-zinc-200"
                : "border-transparent text-zinc-500 hover:text-zinc-400"
            }`}
          >
            {session?.status === "running" ? (
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_6px_2px_rgba(52,211,153,0.5)] animate-pulse" aria-hidden />
            ) : (
              <span className="flex h-2 w-2 rounded-full bg-zinc-600" aria-hidden />
            )}
            Runtime output
            {!isOpen && hasErrors && (
              <span className="rounded-full bg-red-950/60 px-1.5 py-0.5 text-[10px] text-red-400">err</span>
            )}
            {!isOpen && (
              <span className="text-zinc-600 text-[10px] normal-case tracking-normal">
                {logsLoading ? "loading…" : `${logs.length}`}
              </span>
            )}
          </button>

          {/* Agent tab */}
          <button
            type="button"
            onClick={() => { setActiveTab("agent"); if (!isOpen) onToggleOpen(); }}
            className={`flex items-center gap-2 border-b-2 px-4 py-2.5 font-mono text-xs font-semibold uppercase tracking-widest transition ${
              activeTab === "agent" && isOpen
                ? "border-indigo-500 text-zinc-200"
                : "border-transparent text-zinc-500 hover:text-zinc-400"
            }`}
          >
            <span className="flex h-4 w-4 items-center justify-center rounded bg-indigo-600 text-[8px] font-bold text-white">A</span>
            Atlas
            {messages.length > 0 && (
              <span className="rounded-full bg-indigo-950/60 px-1.5 py-0.5 text-[10px] text-indigo-400">
                {messages.length}
              </span>
            )}
          </button>

          {/* Session label */}
          {session && (
            <span className="ml-3 rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-500">
              {session.runName}
            </span>
          )}
        </div>

        {/* Right: controls */}
        <div className="flex items-center gap-2">
          {activeTab === "logs" && isOpen && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); onToggleFollow(); }}
              title={consoleFollow ? "Following — click to stop" : "Not following — click to follow"}
              className={`rounded-full border px-2.5 py-0.5 font-mono text-[10px] uppercase tracking-widest transition ${
                consoleFollow
                  ? "border-emerald-700 bg-emerald-950/50 text-emerald-400"
                  : "border-zinc-700 bg-zinc-900 text-zinc-500 hover:border-zinc-500"
              }`}
            >
              Follow
            </button>
          )}
          {activeTab === "agent" && isOpen && (
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); clearHistory(); }}
              className="rounded-lg p-1.5 text-zinc-600 transition hover:bg-zinc-800 hover:text-zinc-400"
              aria-label="Clear agent"
              title="clear"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-2 14H7L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" /></svg>
            </button>
          )}
          <button
            type="button"
            onClick={onToggleOpen}
            className="rounded-lg p-1 text-zinc-500 transition hover:text-zinc-300"
            aria-label={isOpen ? "Collapse panel" : "Expand panel"}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none"
              stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
              className={`transition-transform ${isOpen ? "rotate-180" : ""}`} aria-hidden
            >
              <path d="m18 15-6-6-6 6" />
            </svg>
          </button>
        </div>
      </div>

      {/* Panel body */}
      {isOpen && (
        <div className="bg-zinc-950">
          {/* ── Logs tab ── */}
          {activeTab === "logs" && (
            <div
              ref={consoleBodyRef}
              style={{ height: consoleHeight }}
              className="overflow-y-auto px-5 py-3 font-mono text-xs"
              onScroll={onConsoleScroll}
            >
              {!session ? (
                <p className="text-zinc-600">Select a project to inspect live session logs.</p>
              ) : logsLoading ? (
                <p className="text-zinc-600 animate-pulse">Loading logs…</p>
              ) : logs.length === 0 ? (
                <p className="text-zinc-600">No logs yet.</p>
              ) : (
                logs.map((log) => (
                  <div key={log.id} className="flex flex-wrap gap-2 py-0.5">
                    <span className="text-zinc-600">{new Date(log.ts).toLocaleTimeString()}</span>
                    <span className={`font-semibold ${
                      log.level === "ERROR" || log.kind === "error" ? "text-red-400"
                        : log.level === "WARN" ? "text-amber-400"
                        : "text-emerald-400"
                    }`}>{log.level}</span>
                    <span className="text-zinc-500">{log.module}:{log.lineno}</span>
                    <span className="text-zinc-200">{log.msg}</span>
                  </div>
                ))
              )}
            </div>
          )}

          {/* ── Agent tab ── */}
          {activeTab === "agent" && (
            <div className="flex flex-col" onClick={focusInput}>
              {/* Belief bar */}
              {beliefState && <BeliefStatusBar belief={beliefState} />}

              {/* Agent output */}
              <div
                ref={agentScrollRef}
                style={{ height: consoleHeight }}
                className="overflow-y-auto px-5 py-3 font-mono text-xs"
              >
                {messages.length === 0 && !isAgentLoading ? (
                  <div className="space-y-3 py-4">
                    <div className="text-zinc-500 leading-relaxed">
                      <p className="text-indigo-400/80 font-semibold">atlas v1.0</p>
                      <p className="mt-1">
                        Adaptive training diagnostics agent.{" "}
                        {noSession ? (
                          <span className="text-amber-400/70">Select a session to start.</span>
                        ) : (
                          <span>Type a question or use a quick command below.</span>
                        )}
                      </p>
                      <p className="mt-1 text-zinc-600">
                        Type <span className="text-zinc-400">clear</span> to reset.
                        Use <span className="text-zinc-400">↑/↓</span> for history.
                      </p>
                    </div>
                    {!noSession && (
                      <div className="flex flex-wrap gap-2 pt-1">
                        {QUICK_ACTIONS.map((qa) => (
                          <button
                            key={qa.label}
                            type="button"
                            onClick={(e) => { e.stopPropagation(); handleQuickAction(qa.prompt); }}
                            className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-2.5 py-1.5 font-mono text-[11px] text-zinc-400 transition hover:border-zinc-600 hover:bg-zinc-800/80 hover:text-zinc-200"
                          >
                            /{qa.label}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-1">
                    {messages.map((msg) => <TerminalEntry key={msg.id} msg={msg} />)}
                    {isAgentLoading && (
                      <div className="flex items-center gap-2 py-1.5 pl-4">
                        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-[8px] font-bold text-white">A</span>
                        <div className="flex gap-1">
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-600" style={{ animationDelay: "0ms" }} />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-600" style={{ animationDelay: "150ms" }} />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-600" style={{ animationDelay: "300ms" }} />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Error banner */}
              {agentError && (
                <div className="border-t border-red-900/40 bg-red-950/20 px-5 py-1.5 font-mono text-[11px] text-red-400">
                  <span className="text-red-600">err:</span> {agentError}
                </div>
              )}

              {/* Agent input prompt */}
              <form
                onSubmit={handleSubmit}
                className="flex items-center gap-2 border-t border-zinc-800 bg-zinc-900/40 px-5 py-2.5"
              >
                <span className="shrink-0 select-none font-mono text-xs font-bold text-indigo-400">$</span>
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={noSession ? "select a session first…" : "ask atlas…"}
                  disabled={noSession || isAgentLoading}
                  autoComplete="off"
                  spellCheck={false}
                  className="flex-1 bg-transparent font-mono text-xs text-zinc-100 placeholder-zinc-600 outline-none caret-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
                />
                {isAgentLoading && (
                  <span className="shrink-0 font-mono text-[10px] text-zinc-600 animate-pulse">thinking…</span>
                )}
              </form>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export {
  BottomTerminalPanel,
  ModelPanel, SAMPLE_LOGS,
  SAMPLE_MODEL,
  SAMPLE_SESSIONS, SessionIssuesPanel,
  SessionList,
  SessionLogList,
  TrainSessionPanel,
  TrainStepList
};

  export type { SessionLog, TrainSession, TrainStep };

