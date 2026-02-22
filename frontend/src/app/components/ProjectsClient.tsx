"use client";

import type { TrainSession as ApiTrainSession, Project } from "@/lib/client";
import {
  createProjectProjectsPostMutation,
  getModelSessionsSessionIdModelGetOptions,
  getProjectsProjectsGetOptions,
  getProjectsProjectsGetQueryKey,
  getSessionHealthDiagnosticsSessionsSessionIdHealthGetOptions,
  getSessionLogsSessionsSessionIdLogsGetOptions,
  getStepsSessionsSessionIdStepGetOptions,
  getTrainSessionsSessionsProjectProjectIdGetOptions,
} from "@/lib/client/@tanstack/react-query.gen";
import { useEventSource } from "@/lib/use-event-source";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import type { TrainSession as PanelTrainSession } from "./ProjectTrainingPanels";
import {
  ModelPanel,
  SessionIssuesPanel,
  SessionList,
  SessionLogList,
  TrainSessionPanel,
  TrainStepList,
} from "./ProjectTrainingPanels";
import ProjectTrendChart from "./ProjectTrendChart";

function mapApiSessionToPanel(api: ApiTrainSession): PanelTrainSession {
  return {
    id: api.id ?? 0,
    projectId: api.project_id,
    runId: api.run_id,
    runName: api.run_name,
    startedAt: api.started_at,
    endedAt: api.ended_at ?? null,
    device: api.device ?? "",
    cudaAvailable: api.cuda_available ?? false,
    pytorchVersion: api.pytorch_version ?? "",
    config: api.config ?? null,
    summary: api.summary ?? null,
    status: (api.status ?? "pending") as PanelTrainSession["status"],
  };
}

import AgentChatPanel from "./AgentChatPanel";
import ProactiveInsightBanner, {
  useProactiveInsights,
  type InsightItem,
} from "./ProactiveInsightBanner";
import SustainabilityPanel from "./SustainabilityPanel";
import ThreeScene from "./ThreeScene";

const SELECTED_PROJECT_ID_KEY = "atlas-selected-project-id";

type ProjectsClientProps = {
  fontClassName: string;
};

const SUGGESTIONS = [
  "Cache immutable assets with long-lived headers and hash-based filenames.",
  "Split routes by intent and defer non-critical bundles with dynamic imports.",
  "Batch API requests per screen and prefetch the next likely view during idle time.",
  "Compress and resize imagery at build time; serve modern formats with fallbacks.",
  "Instrument core flows to surface long tasks, then offload heavy work to workers.",
] as const;

function getStoredProjectId(): number | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(SELECTED_PROJECT_ID_KEY);
  if (!raw) return null;
  const id = Number(raw);
  return Number.isNaN(id) ? null : id;
}

function setStoredProjectId(id: number | null) {
  if (typeof window === "undefined") return;
  if (id == null) {
    window.localStorage.removeItem(SELECTED_PROJECT_ID_KEY);
  } else {
    window.localStorage.setItem(SELECTED_PROJECT_ID_KEY, String(id));
  }
}

function getRelativeRefreshLabel(at: Date, now: Date): string {
  const sec = Math.floor((now.getTime() - at.getTime()) / 1000);
  if (sec < 10) return "A few moments ago";
  if (sec < 60) return "Less than a minute ago";
  const min = Math.floor(sec / 60);
  if (min === 1) return "A minute ago";
  if (min < 60) return `${min} minutes ago`;
  const hr = Math.floor(min / 60);
  if (hr === 1) return "An hour ago";
  if (hr < 24) return `${hr} hours ago`;
  const day = Math.floor(hr / 24);
  if (day === 1) return "A day ago";
  return `${day} days ago`;
}

export default function ProjectsClient({
  fontClassName,
}: ProjectsClientProps) {
  const queryClient = useQueryClient();
  const { insights, addInsight, dismiss: dismissInsight } =
    useProactiveInsights();

  const handleInsightEvent = (eventType: string, data: Record<string, unknown>) => {
    if (eventType === "agent.insight") {
      addInsight(data as { severity?: string; title?: string; body?: string; is_revision?: boolean });
    }
  };

  useEventSource({
    projectId: null,
    enabled: true,
    onEvent: handleInsightEvent,
  });
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(
    getStoredProjectId
  );
  const [isProjectsOpen, setIsProjectsOpen] = useState(true);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(
    null
  );
  const [lastRefreshAt, setLastRefreshAt] = useState<Date | null>(null);
  const [, setTick] = useState(0);

  // Tick every 30s so "last refresh" label updates
  useEffect(() => {
    if (lastRefreshAt == null) return;
    const id = setInterval(() => setTick((t) => t + 1), 30_000);
    return () => clearInterval(id);
  }, [lastRefreshAt]);

  const {
    data: projects = [],
    isLoading: isProjectsLoading,
    isError: isProjectsError,
    error: projectsError,
  } = useQuery(getProjectsProjectsGetOptions());

  const {
    data: apiSessions = [],
    isLoading: isSessionsLoading,
  } = useQuery({
    ...getTrainSessionsSessionsProjectProjectIdGetOptions({
      path: { project_id: selectedProjectId ?? 0 },
    }),
    enabled: selectedProjectId != null,
  });

  const sessionsForProject = useMemo(
    () => apiSessions.map(mapApiSessionToPanel),
    [apiSessions]
  );

  const createProjectMutation = useMutation({
    ...createProjectProjectsPostMutation(),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: getProjectsProjectsGetQueryKey() });
      const id = created?.id ?? null;
      if (id != null) {
        setSelectedProjectId(id);
        setStoredProjectId(id);
      }
    },
  });

  // Persist selection to localStorage when it changes
  const handleSelectProject = (id: number) => {
    setSelectedProjectId(id);
    setStoredProjectId(id);
  };

  const selectedProject = useMemo(
    () =>
      selectedProjectId == null
        ? null
        : projects.find((p) => p.id === selectedProjectId) ?? null,
    [projects, selectedProjectId]
  );

  // Clear selection if selected project no longer exists in the list
  useEffect(() => {
    if (selectedProjectId == null) return;
    const exists = projects.some((p) => p.id === selectedProjectId);
    if (!exists) {
      setSelectedProjectId(null);
      setStoredProjectId(null);
    }
  }, [projects, selectedProjectId]);

  // Auto-select newest session (first in list, backend returns newest-first).
  // When a new run starts, it appears as first → we select it so logs/steps show the active run.
  useEffect(() => {
    if (!selectedProjectId || sessionsForProject.length === 0) {
      if (!selectedProjectId) setSelectedSessionId(null);
      return;
    }
    const newest = sessionsForProject[0];
    const currentInList = selectedSessionId != null && sessionsForProject.some((s) => s.id === selectedSessionId);
    const shouldSelectNewest =
      newest != null &&
      (selectedSessionId == null || !currentInList || newest.id > selectedSessionId);
    if (shouldSelectNewest) {
      setSelectedSessionId(newest.id);
    }
  }, [selectedProjectId, sessionsForProject, selectedSessionId]);

  const activeSession = useMemo(
    () =>
      selectedProject && sessionsForProject.length > 0
        ? sessionsForProject.find((s) => s.id === selectedSessionId) ??
          sessionsForProject[0] ??
          null
        : null,
    [selectedProject, sessionsForProject, selectedSessionId]
  );

  const sessionIdForModel = activeSession?.id ?? null;
  const {
    data: apiModel,
    isLoading: isModelLoading,
    isError: isModelError,
  } = useQuery({
    ...getModelSessionsSessionIdModelGetOptions({
      path: { session_id: sessionIdForModel ?? 0 },
    }),
    enabled: sessionIdForModel != null,
  });

  const modelForPanel =
    sessionIdForModel == null
      ? null
      : isModelLoading
        ? undefined
        : isModelError || apiModel == null
          ? null
          : apiModel;

  const {
    data: apiSteps = [],
    isLoading: isStepsLoading,
  } = useQuery({
    ...getStepsSessionsSessionIdStepGetOptions({
      path: { session_id: sessionIdForModel ?? 0 },
    }),
    enabled: sessionIdForModel != null,
  });

  const {
    data: apiLogs = [],
    isLoading: isLogsLoading,
  } = useQuery({
    ...getSessionLogsSessionsSessionIdLogsGetOptions({
      path: { session_id: sessionIdForModel ?? 0 },
    }),
    enabled: sessionIdForModel != null,
    refetchInterval:
      activeSession?.status === "running" || activeSession?.status === "analyzing"
        ? 30_000
        : false,
  });

  const {
    data: healthData,
    isLoading: isHealthLoading,
  } = useQuery({
    ...getSessionHealthDiagnosticsSessionsSessionIdHealthGetOptions({
      path: { session_id: sessionIdForModel ?? 0 },
    }),
    enabled: sessionIdForModel != null && activeSession?.status == "pending",
  });

  const logsForPanel = useMemo(() => {
    return apiLogs.map((log) => ({
      id: log.id ?? 0,
      sessionId: log.session_id,
      ts: log.ts,
      level: log.level,
      msg: log.msg,
      module: log.module ?? "",
      lineno: log.lineno ?? 0,
      kind: (log.kind ?? "console") as "console" | "error",
    }));
  }, [apiLogs]);

  const handleNewProject = () => {
    const nextNumber = projects.length + 1;
    createProjectMutation.mutate({
      body: { name: `Project ${nextNumber}` },
    });
  };

  const handleSuggestionClick = async (suggestion: string) => {
    if (!selectedProject) return;

    try {
      await fetch("/api/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          suggestion,
          token: String(selectedProject.id),
          projectName: selectedProject.name,
        }),
      });
    } catch (error) {
      console.error("Failed to send suggestion", error);
    }
  };

  const handleRefreshAll = () => {
    queryClient.invalidateQueries();
    setLastRefreshAt(new Date());
  };

  const lastRefreshLabel =
    lastRefreshAt == null
      ? null
      : getRelativeRefreshLabel(lastRefreshAt, new Date());


  return (
    <div className={`${fontClassName} min-h-screen bg-zinc-900 text-zinc-100`}>
      <div className="group fixed bottom-6 right-6 z-50 flex items-center gap-2">
        {lastRefreshLabel ? (
          <span className="text-xs text-zinc-500 opacity-0 transition-opacity group-hover:opacity-100">
            {lastRefreshLabel}
          </span>
        ) : null}
        <button
          type="button"
          onClick={handleRefreshAll}
          aria-label="Refresh all data"
          className="flex h-12 w-12 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-200 shadow-lg shadow-black/30 transition hover:border-zinc-500 hover:bg-zinc-700 hover:text-white focus:outline-none"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden
          >
            <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
            <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
            <path d="M16 21h5v-5" />
          </svg>
        </button>
      </div>
      <div className="relative isolate overflow-hidden">
        <header className="mx-auto flex w-full max-w-[1700px] flex-wrap items-center justify-between gap-4 px-6 pt-10">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-sm font-bold uppercase text-zinc-900">
              A
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-400">
                Atlas Workspace
              </p>
              <h1 className="text-xl font-semibold">Projects</h1>
            </div>
          </div>

          <div className="flex flex-1 items-center justify-end gap-3">
            <button
              type="button"
              onClick={handleNewProject}
              disabled={createProjectMutation.isPending}
              className="rounded-full bg-white px-5 py-2 text-sm font-semibold text-zinc-900 shadow-lg shadow-white/10 disabled:opacity-60"
            >
              {createProjectMutation.isPending ? "Creating…" : "New project"}
            </button>
          </div>
        </header>
        <div className="mx-auto w-full max-w-[1700px] px-6 pb-12 pt-10">
          {/* Proactive insight banners */}
          <div className="mb-4">
            <ProactiveInsightBanner
              insights={insights}
              onDismiss={dismissInsight}
              onAskAtlas={(insight: InsightItem) => {
                // Open agent chat is handled by the AgentChatPanel toggle
                // For now, insights are clickable
              }}
            />
          </div>
          <div className="grid gap-6 xl:grid-cols-12">
            <div className="space-y-6 xl:col-span-3">
            <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                    Projects
                  </p>
                  <h2 className="text-lg font-semibold">Your projects</h2>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-zinc-500">
                    {isProjectsLoading
                      ? "…"
                      : isProjectsError
                        ? "Error"
                        : `${projects.length} total`}
                  </span>
                  <button
                    type="button"
                    onClick={() => setIsProjectsOpen((prev) => !prev)}
                    aria-expanded={isProjectsOpen}
                    className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-300 transition hover:border-zinc-600"
                  >
                    {isProjectsOpen ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              {isProjectsOpen ? (
                isProjectsError ? (
                  <div className="mt-4 rounded-2xl border border-red-900/50 bg-red-950/20 px-4 py-5 text-sm text-red-300">
                    Failed to load projects:{" "}
                    {projectsError instanceof Error
                      ? projectsError.message
                      : "Unknown error"}
                  </div>
                ) : projects.length === 0 && !isProjectsLoading ? (
                  <div className="mt-4 rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/40 px-4 py-5 text-sm text-zinc-400">
                    No projects yet. Click "New project" to create one.
                  </div>
                ) : (
                  <div className="mt-4 grid gap-3">
                    {isProjectsLoading ? (
                      <div className="py-4 text-center text-sm text-zinc-500">
                        Loading projects…
                      </div>
                    ) : (
                      projects.map((project) => (
                        <button
                          type="button"
                          key={project.id ?? project.name}
                          onClick={() =>
                            project.id != null && handleSelectProject(project.id)
                          }
                          aria-pressed={
                            selectedProjectId === project.id
                          }
                          className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-left transition ${
                            selectedProjectId === project.id
                              ? "border-white/70 bg-white/10 shadow-lg shadow-white/10"
                              : "border-zinc-800 bg-zinc-900/60 hover:border-zinc-600"
                          }`}
                        >
                          <div>
                            <p className="text-sm font-semibold text-zinc-100">
                              {project.name}
                            </p>
                            <p className="text-xs text-zinc-500">
                              Created{" "}
                              {project.created_at
                                ? new Date(
                                    project.created_at
                                  ).toLocaleString()
                                : "—"}
                            </p>
                          </div>
                          <div className="flex items-center gap-2 text-xs font-medium">
                            {"status" in project && project.status != null ? (
                              <span className="flex items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-900/80 px-2.5 py-1 text-zinc-300">
                                {project.status === "pending" ? (
                                  <span className="text-[10px] font-medium uppercase tracking-wider">
                                    PENDING ACTION
                                  </span>
                                ) : project.status === "running" ? (
                                  <>
                                    <span
                                      className="h-2 w-2 shrink-0 rounded-full bg-emerald-400 shadow-[0_0_10px_4px_rgba(52,211,153,0.5)] animate-pulse"
                                      aria-hidden
                                    />
                                    <span className="text-[10px] font-medium uppercase tracking-wider text-emerald-400">
                                      running
                                    </span>
                                  </>
                                ) : (
                                  <span className="text-[10px] uppercase tracking-[0.15em]">
                                    {project.status}
                                  </span>
                                )}
                              </span>
                            ) : null}
                            {selectedProjectId === project.id ? (
                              <span className="rounded-full bg-white/15 px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-white/80">
                                Selected
                              </span>
                            ) : null}
                            {project.id != null ? (
                              <span className="rounded-full bg-zinc-950 px-3 py-1 text-zinc-300">
                                <span className="mr-2 text-zinc-500">ID</span>
                                <span className="font-mono tracking-tight">
                                  {project.id}
                                </span>
                              </span>
                            ) : null}
                          </div>
                        </button>
                      ))
                    )}
                  </div>
                )
              ) : (
                <p className="mt-4 text-xs text-zinc-500">
                  Project list hidden.
                </p>
              )}
            </section>
            <PipelineSuggestions
              selectedProject={selectedProject}
              onSuggestionClick={handleSuggestionClick}
            />
            <ProjectTrendChart projectId={selectedProjectId} />
            </div>
            <div className="space-y-6 xl:col-span-5">
            <ThreeScene model={modelForPanel ?? null} />
            <SessionList
              selectedProject={selectedProject}
              sessions={sessionsForProject}
              sessionsLoading={isSessionsLoading}
              selectedSessionId={selectedSessionId}
              onSelectSession={setSelectedSessionId}
            />
            <TrainSessionPanel session={activeSession} />
            </div>
            <div className="space-y-6 xl:col-span-4">
            <ModelPanel session={activeSession} model={modelForPanel} />
            <TrainStepList
              session={activeSession}
              steps={apiSteps}
              stepsLoading={isStepsLoading}
            />
            <SessionLogList session={activeSession} logs={logsForPanel} logsLoading={isLogsLoading} />
            <SessionIssuesPanel
              session={activeSession}
              health={healthData ?? null}
              healthLoading={isHealthLoading}
            />
            <SustainabilityPanel sessionId={selectedSessionId} />
            </div>
          </div>
        </div>
      </div>
      {/* Agent chat panel */}
      <AgentChatPanel
        sessionId={selectedSessionId}
        projectId={selectedProjectId}
      />
    </div>
  );
}

type PipelineSuggestionsProps = {
  selectedProject: Project | null;
  onSuggestionClick: (suggestion: string) => void;
};

function PipelineSuggestions({
  selectedProject,
  onSuggestionClick,
}: PipelineSuggestionsProps) {
  const isDisabled = !selectedProject;
  return (
    <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
            Optimisation hints
          </p>
          <h2 className="text-lg font-semibold">Pipeline suggestions</h2>
        </div>
        <span className="rounded-full border border-zinc-800 bg-zinc-900/60 px-3 py-1 text-xs text-zinc-400">
          Frontend focus
        </span>
      </div>
      <div className="mt-4 grid gap-3 text-sm text-zinc-200">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSuggestionClick(suggestion)}
            disabled={isDisabled}
            className={`rounded-2xl border px-4 py-3 text-left transition ${
              isDisabled
                ? "cursor-not-allowed border-zinc-900/80 bg-zinc-900/30 text-zinc-600"
                : "border-zinc-800 bg-zinc-900/50 hover:border-zinc-600 hover:bg-zinc-900/70"
            }`}
          >
            {suggestion}
          </button>
        ))}
      </div>
      {isDisabled ? (
        <p className="mt-3 text-xs text-zinc-500">
          Select a project to send suggestions.
        </p>
      ) : null}
    </section>
  );
}
