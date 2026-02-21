"use client";

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
  status: "running" | "completed" | "failed" | "paused" | "pending";
};

type TrainModel = {
  id: number;
  sessionId: number;
  architecture: Record<string, unknown>;
  hyperparameters: Record<string, unknown>;
};

type TrainStep = {
  id: number;
  sessionId: number;
  epochIndex: number;
  timestamp: string;
  durationSeconds: number;
  payload: Record<string, unknown>;
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

const SAMPLE_STEPS: TrainStep[] = [
  {
    id: 101,
    sessionId: 17,
    epochIndex: 12,
    timestamp: "2026-02-21T08:04:12Z",
    durationSeconds: 43.2,
    payload: { train_loss: 0.392, val_loss: 0.344, val_acc: 0.913 },
  },
  {
    id: 102,
    sessionId: 17,
    epochIndex: 13,
    timestamp: "2026-02-21T08:05:01Z",
    durationSeconds: 42.7,
    payload: { train_loss: 0.361, val_loss: 0.332, val_acc: 0.917 },
  },
  {
    id: 103,
    sessionId: 17,
    epochIndex: 14,
    timestamp: "2026-02-21T08:05:45Z",
    durationSeconds: 43.5,
    payload: { train_loss: 0.348, val_loss: 0.326, val_acc: 0.919 },
  },
];

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
  selectedProject: { token: string } | null;
  sessions: TrainSession[];
  selectedSessionId: number | null;
  onSelectSession: (sessionId: number) => void;
};

function SessionList({
  selectedProject,
  sessions,
  selectedSessionId,
  onSelectSession,
}: SessionListProps) {
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
          {sessions.length} runs
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
};

function TrainSessionPanel({ session }: TrainSessionPanelProps) {
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
          <div className="grid gap-3 md:grid-cols-2">
            <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                Config
              </p>
              <div className="mt-3 grid gap-2">
                {session.config ? (
                  Object.entries(session.config).map(([key, value]) => (
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
                    No config captured.
                  </p>
                )}
              </div>
            </div>
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
                )}
              </div>
            </div>
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
  model: TrainModel;
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
      {session ? (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200 md:grid-cols-2">
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4">
            <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
              Architecture
            </p>
            <div className="mt-3 grid gap-2">
              {Object.entries(model.architecture).map(([key, value]) => (
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
              {Object.entries(model.hyperparameters).map(([key, value]) => (
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
      ) : (
        <p className="mt-4 text-xs text-zinc-500">
          Select a project to inspect the bound model definition.
        </p>
      )}
    </section>
  );
}

type TrainStepListProps = {
  session: TrainSession | null;
  steps: TrainStep[];
};

function TrainStepList({ session, steps }: TrainStepListProps) {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Train steps
          </p>
          <h2 className="text-lg font-semibold">Recent epochs</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {steps.length} steps
        </span>
      </div>
      {session ? (
        <div className="mt-4 grid gap-3 text-sm text-zinc-200">
          {steps.map((step) => (
            <div
              key={step.id}
              className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-4"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm font-semibold text-zinc-100">
                  Epoch {step.epochIndex}
                </p>
                <span className="text-xs text-zinc-500">
                  {step.durationSeconds.toFixed(1)}s
                </span>
              </div>
              <div className="mt-2 flex flex-wrap gap-3 text-xs text-zinc-400">
                <span>{new Date(step.timestamp).toLocaleString()}</span>
                <span>Payload</span>
              </div>
              <div className="mt-3 grid gap-2">
                {Object.entries(step.payload).map(([key, value]) => (
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
          ))}
        </div>
      ) : (
        <p className="mt-4 text-xs text-zinc-500">
          Select a project to review recent training steps.
        </p>
      )}
    </section>
  );
}

type SessionLogListProps = {
  session: TrainSession | null;
  logs: SessionLog[];
};

function SessionLogList({ session, logs }: SessionLogListProps) {
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Session logs
          </p>
          <h2 className="text-lg font-semibold">Runtime output</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          {logs.length} entries
        </span>
      </div>
      {session ? (
        <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-950/80">
          <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-2 text-xs text-zinc-500">
            <span className="uppercase tracking-[0.2em]">Console</span>
            <span>{logs.length} lines</span>
          </div>
          <div className="max-h-80 overflow-y-auto px-4 py-3 font-mono text-xs text-zinc-200">
            {logs.map((log) => (
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
            ))}
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

export {
  SAMPLE_LOGS,
  SAMPLE_MODEL,
  SAMPLE_SESSIONS,
  SAMPLE_STEPS,
  ModelPanel,
  SessionList,
  SessionLogList,
  TrainSessionPanel,
  TrainStepList,
};

export type { SessionLog, TrainModel, TrainSession, TrainStep };
