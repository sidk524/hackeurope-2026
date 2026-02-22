"use client";

import type { TrainSession as ApiTrainSession } from "@/lib/client";
import {
  createProjectProjectsPostMutation,
  getModelSessionsSessionIdModelGetOptions,
  getProjectsProjectsGetOptions,
  getProjectsProjectsGetQueryKey,
  getSessionHealthDiagnosticsSessionsSessionIdHealthGetOptions,
  getSessionLogsSessionsSessionIdLogsGetOptions,
  getStepsSessionsSessionIdStepGetOptions,
  getTrainSessionsSessionsProjectProjectIdGetOptions,
  getTrainSessionsSessionsProjectProjectIdGetQueryKey,
  sessionActionSessionsSessionIdActionPostMutation
} from "@/lib/client/@tanstack/react-query.gen";
import { useEventSource } from "@/lib/use-event-source";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { TrainSession as PanelTrainSession } from "./ProjectTrainingPanels";
import {
  BottomTerminalPanel,
  ModelPanel,
  SessionIssuesPanel,
  TrainSessionPanel,
  TrainStepList
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

import ProactiveInsightBanner, {
  useProactiveInsights,
  type InsightItem,
} from "./ProactiveInsightBanner";
import SustainabilityPanel from "./SustainabilityPanel";
import ThreeScene from "./ThreeScene";

const SELECTED_PROJECT_ID_KEY = "atlas-selected-project-id";

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function toFiniteNumber(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function severityWeight(severity: unknown): number {
  if (severity === "critical") return 1;
  if (severity === "warning") return 0.65;
  if (severity === "info") return 0.35;
  return 0.5;
}

function metricPenalty(metricKey: unknown, metricValue: unknown): number {
  const key = typeof metricKey === "string" ? metricKey : "";
  const mv = metricValue && typeof metricValue === "object"
    ? (metricValue as Record<string, unknown>)
    : {};

  if (key === "is_dead") return 1;
  if (key === "gradient_norm_mean") return 0.75;
  if (key === "activation_var_of_means" || key === "activation_std") return 0.7;
  if (key === "activation_correlation") {
    const corr = Math.abs(toFiniteNumber(mv.avg_correlation) ?? 0);
    return clamp01((corr - 0.9) / 0.1);
  }
  if (key === "weight_sparsity") {
    const sparsity = toFiniteNumber(mv.weight_sparsity) ?? 0;
    return clamp01((sparsity - 0.4) / 0.6);
  }
  if (key === "compute_to_param_ratio" || key === "param_to_compute_ratio") {
    const ratio = toFiniteNumber(mv.ratio) ?? 0;
    return clamp01((ratio - 3) / 12);
  }
  if (key === "pct_total" || key === "avg_pct_total" || key === "pct_of_total") {
    const pct = toFiniteNumber(mv.pct_total) ?? toFiniteNumber(mv.avg_pct) ?? toFiniteNumber(mv.pct) ?? 0;
    return clamp01((pct - 20) / 50);
  }
  if (key === "out_channels" || key === "kernel_size") return 0.45;
  return 0.5;
}

function linkedLayerIds(issue: {
  layer_id?: unknown;
  metric_key?: unknown;
  metric_value?: unknown;
}): string[] {
  const ids = new Set<string>();
  if (typeof issue.layer_id === "string" && issue.layer_id.length > 0) {
    ids.add(issue.layer_id);
  }
  if (issue.metric_key === "activation_correlation" && issue.metric_value && typeof issue.metric_value === "object") {
    const mv = issue.metric_value as Record<string, unknown>;
    if (typeof mv.layer_a === "string" && mv.layer_a.length > 0) ids.add(mv.layer_a);
    if (typeof mv.layer_b === "string" && mv.layer_b.length > 0) ids.add(mv.layer_b);
  }
  return Array.from(ids);
}

type ProjectsClientProps = {
  fontClassName: string;
};

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
  const [projectDropdownOpen, setProjectDropdownOpen] = useState(false);
  const [sessionDropdownOpen, setSessionDropdownOpen] = useState(false);
  const [newProjectHover, setNewProjectHover] = useState(false);
  const [newProjectInputFocused, setNewProjectInputFocused] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const projectDropdownRef = useRef<HTMLDivElement>(null);
  const sessionDropdownRef = useRef<HTMLDivElement>(null);
  const newProjectInputRef = useRef<HTMLInputElement>(null);
  const [isConsoleOpen, setIsConsoleOpen] = useState(false);
  const [consoleHeight, setConsoleHeight] = useState(208); // px
  const [consoleFollow, setConsoleFollow] = useState(true);
  const [panelHeightPx, setPanelHeightPx] = useState(0);
  const consoleBodyRef = useRef<HTMLDivElement>(null);

  const handlePanelHeightChange = useCallback((heightPx: number) => {
    setPanelHeightPx(heightPx);
  }, []);
  const isDragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartHeight = useRef(0);

  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;
    dragStartY.current = e.clientY;
    dragStartHeight.current = consoleHeight;
    document.body.style.cursor = "ns-resize";
    document.body.style.userSelect = "none";

    const onMove = (mv: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = dragStartY.current - mv.clientY;
      setConsoleHeight(Math.min(600, Math.max(80, dragStartHeight.current + delta)));
    };
    const onUp = () => {
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [consoleHeight]);

  const [claudeReply, setClaudeReply] = useState<string>("");
  const [claudeError, setClaudeError] = useState<string>("");
  const [isClaudeLoading, setIsClaudeLoading] = useState(false);
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

  const sessionActionMutation = useMutation({
    ...sessionActionSessionsSessionIdActionPostMutation(),
    onSuccess: (_data, variables) => {
      if (selectedProjectId != null) {
        queryClient.invalidateQueries({
          queryKey: getTrainSessionsSessionsProjectProjectIdGetQueryKey({
            path: { project_id: selectedProjectId },
          }),
        });
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

  // Auto-select newest session only when there is no selection or the selected session is no longer in the list.
  // Do not overwrite user's explicit choice when they pick an older session.
  useEffect(() => {
    if (!selectedProjectId || sessionsForProject.length === 0) {
      if (!selectedProjectId) setSelectedSessionId(null);
      return;
    }
    const newest = sessionsForProject[0];
    const currentInList = selectedSessionId != null && sessionsForProject.some((s) => s.id === selectedSessionId);
    const shouldSelectNewest =
      newest != null && (selectedSessionId == null || !currentInList);
    if (shouldSelectNewest) {
      setSelectedSessionId(newest.id);
    }
  }, [selectedProjectId, sessionsForProject, selectedSessionId]);

  // Close header dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (projectDropdownRef.current?.contains(target) || sessionDropdownRef.current?.contains(target)) return;
      setProjectDropdownOpen(false);
      setSessionDropdownOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

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
    enabled: sessionIdForModel != null,
  });

  const sustainabilityScores = useMemo(() => {
    const issues = healthData?.issues ?? [];
    const penaltiesByLayer = new Map<string, number[]>();
    for (const issue of issues) {
      const layerIds = linkedLayerIds(issue as { layer_id?: unknown; metric_key?: unknown; metric_value?: unknown });
      if (layerIds.length === 0) continue;
      const weight = severityWeight((issue as { severity?: unknown }).severity);
      const penalty = metricPenalty(
        (issue as { metric_key?: unknown }).metric_key,
        (issue as { metric_value?: unknown }).metric_value
      );
      const combined = clamp01(weight * penalty);
      for (const layerId of layerIds) {
        const current = penaltiesByLayer.get(layerId) ?? [];
        current.push(combined);
        penaltiesByLayer.set(layerId, current);
      }
    }

    const scores: Record<string, number> = {};
    for (const [layerId, penalties] of penaltiesByLayer.entries()) {
      let unsustainability = 0;
      for (const p of penalties) {
        unsustainability = unsustainability + p * (1 - unsustainability) * 0.9;
      }
      scores[layerId] = Math.max(0, Math.min(100, Math.round((1 - unsustainability) * 100)));
    }
    return scores;
  }, [healthData]);

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

  // Auto-scroll to bottom when new logs arrive and follow is on
  useEffect(() => {
    if (!consoleFollow || !consoleBodyRef.current) return;
    consoleBodyRef.current.scrollTop = consoleBodyRef.current.scrollHeight;
  }, [logsForPanel, consoleFollow]);

  const isNewProjectExpanded = newProjectHover || newProjectInputFocused;

  const handleNewProject = (name?: string) => {
    const nextNumber = projects.length + 1;
    const projectName =
      (name ?? newProjectName)?.trim() || `Project ${nextNumber}`;
    createProjectMutation.mutate(
      { body: { name: projectName } },
      {
        onSuccess: () => {
          setNewProjectName("");
          setNewProjectInputFocused(false);
        },
      }
    );
  };

  const handleRefreshAll = () => {
    queryClient.invalidateQueries();
    setLastRefreshAt(new Date());
  };

  const lastRefreshLabel =
    lastRefreshAt == null
      ? null
      : getRelativeRefreshLabel(lastRefreshAt, new Date());


  // Button sits just above the console panel; use measured height when available
  const BUTTON_GAP_PX = 12;
  const CONSOLE_CHROME_PX = 40;
  const fallbackBottomPx = isConsoleOpen
    ? CONSOLE_CHROME_PX + consoleHeight + BUTTON_GAP_PX
    : CONSOLE_CHROME_PX + BUTTON_GAP_PX;
  const refreshBottomPx = panelHeightPx > 0 ? panelHeightPx + BUTTON_GAP_PX : fallbackBottomPx;

  return (
    <div className={`${fontClassName} min-h-screen bg-zinc-900 text-zinc-100`} style={{ paddingBottom: isConsoleOpen ? `${consoleHeight + 40}px` : "3rem" }}>
      <div
        className="group fixed right-6 z-60 flex items-center gap-2 transition-[bottom] duration-200"
        style={{ bottom: `${refreshBottomPx}px` }}
      >
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
        <header className="mx-auto flex w-full max-w-[1700px] flex-wrap items-center justify-between gap-4 px-6 pt-6">
          <div className="flex items-center gap-4">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white text-sm font-bold uppercase text-zinc-900">
              A
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.2em] text-zinc-400">
                Atlas Workspace
              </p>
              <h1 className="text-xl font-semibold">Projects</h1>
            </div>
            <div className="h-8 w-px bg-zinc-600" aria-hidden />
            <div className="relative flex flex-col gap-1" ref={projectDropdownRef}>
              <span className="text-[10px] font-medium uppercase tracking-[0.2em] text-zinc-500">
                Project
              </span>
              <button
                type="button"
                onClick={() => {
                  setProjectDropdownOpen((o) => !o);
                  setSessionDropdownOpen(false);
                }}
                aria-expanded={projectDropdownOpen}
                aria-haspopup="listbox"
                className="flex min-w-[200px] items-center justify-between gap-2 rounded-xl border-2 border-zinc-600 bg-zinc-800 px-4 py-3 text-left text-sm font-medium text-white shadow-lg shadow-black/25 transition hover:border-zinc-500 hover:bg-zinc-700 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-white/30 focus:ring-offset-2 focus:ring-offset-zinc-900"
              >
                <span className="truncate">
                  {isProjectsLoading
                    ? "Loading…"
                    : selectedProject?.name ?? "Select project"}
                </span>
                <svg
                  className={`h-4 w-4 shrink-0 text-zinc-400 transition ${projectDropdownOpen ? "rotate-180" : ""}`}
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>
              {projectDropdownOpen ? (
                <div
                  role="listbox"
                  className="absolute left-0 top-full z-50 mt-1.5 max-h-52 min-w-[220px] overflow-y-auto overflow-x-hidden rounded-xl border-2 border-zinc-600 bg-zinc-900 shadow-xl ring-1 ring-black/20"
                >
                  {isProjectsError ? (
                    <div className="px-4 py-3 text-xs text-red-400">
                      Failed to load
                    </div>
                  ) : projects.length === 0 && !isProjectsLoading ? (
                    <div className="px-4 py-3 text-xs text-zinc-500">
                      No projects yet
                    </div>
                  ) : (
                    projects.map((project) => (
                      <button
                        key={project.id ?? project.name}
                        type="button"
                        role="option"
                        aria-selected={selectedProjectId === project.id}
                        onClick={() => {
                          if (project.id != null) {
                            handleSelectProject(project.id);
                            setProjectDropdownOpen(false);
                          }
                        }}
                        className={`flex w-full flex-col gap-0.5 px-4 py-2.5 text-left text-sm transition hover:bg-zinc-800 ${
                          selectedProjectId === project.id
                            ? "bg-zinc-800 text-zinc-100"
                            : "text-zinc-300"
                        }`}
                      >
                        <span className="font-medium">{project.name}</span>
                        {project.id != null ? (
                          <span className="text-xs text-zinc-500">
                            ID {project.id}
                          </span>
                        ) : null}
                      </button>
                    ))
                  )}
                </div>
              ) : null}
            </div>
            <div className="relative flex flex-col gap-1" ref={sessionDropdownRef}>
              <span className="text-[10px] font-medium uppercase tracking-[0.2em] text-zinc-500">
                Session
              </span>
              <button
                type="button"
                onClick={() => {
                  setSessionDropdownOpen((o) => !o);
                  setProjectDropdownOpen(false);
                }}
                disabled={!selectedProjectId}
                aria-expanded={sessionDropdownOpen}
                aria-haspopup="listbox"
                className="flex min-w-[220px] items-center justify-between gap-2 rounded-xl border-2 border-zinc-600 bg-zinc-800 px-4 py-3 text-left text-sm font-medium text-white shadow-lg shadow-black/25 transition hover:border-zinc-500 hover:bg-zinc-700 hover:shadow-xl focus:outline-none focus:ring-2 focus:ring-white/30 focus:ring-offset-2 focus:ring-offset-zinc-900 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <span className="truncate">
                  {!selectedProjectId
                    ? "Select project first"
                    : isSessionsLoading
                      ? "Loading…"
                      : activeSession?.runName ?? "Select session"}
                </span>
                <svg
                  className={`h-4 w-4 shrink-0 text-zinc-400 transition ${sessionDropdownOpen ? "rotate-180" : ""}`}
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden
                >
                  <path d="m6 9 6 6 6-6" />
                </svg>
              </button>
              {sessionDropdownOpen && selectedProjectId ? (
                <div
                  role="listbox"
                  className="absolute left-0 top-full z-50 mt-1.5 max-h-52 min-w-[260px] overflow-y-auto overflow-x-hidden rounded-xl border-2 border-zinc-600 bg-zinc-900 shadow-xl ring-1 ring-black/20"
                >
                  {sessionsForProject.length === 0 && !isSessionsLoading ? (
                    <div className="px-4 py-3 text-xs text-zinc-500">
                      No runs yet
                    </div>
                  ) : (
                    sessionsForProject.map((session) => (
                      <button
                        key={session.id}
                        type="button"
                        role="option"
                        aria-selected={selectedSessionId === session.id}
                        onClick={() => {
                          setSelectedSessionId(session.id);
                          setSessionDropdownOpen(false);
                        }}
                        className={`flex w-full flex-col gap-0.5 px-4 py-2.5 text-left text-sm transition hover:bg-zinc-800 ${
                          selectedSessionId === session.id
                            ? "bg-zinc-800 text-zinc-100"
                            : "text-zinc-300"
                        }`}
                      >
                        <span className="font-medium">{session.runName}</span>
                        <span className="text-xs text-zinc-500">
                          {session.runId} · {session.status}
                        </span>
                      </button>
                    ))
                  )}
                </div>
              ) : null}
            </div>
          </div>

          <div className="flex items-center justify-end gap-3">
            <div
              onMouseEnter={() => setNewProjectHover(true)}
              onMouseLeave={() => setNewProjectHover(false)}
              className="flex overflow-hidden rounded-full bg-white text-zinc-900 shadow-lg shadow-white/10"
            >
              {/* Collapsed: "New project" button — width animates to 0 when expanded */}
              <div
                className="flex shrink-0 overflow-hidden transition-[width] duration-300 ease-out"
                style={{ width: isNewProjectExpanded ? 0 : 130 }}
              >
                <button
                  type="button"
                  onClick={() => {
                    setNewProjectHover(true);
                    setNewProjectInputFocused(true);
                    setTimeout(() => newProjectInputRef.current?.focus(), 50);
                  }}
                  disabled={createProjectMutation.isPending}
                  className="w-[130px] shrink-0 whitespace-nowrap px-5 py-2 text-left text-sm font-semibold disabled:opacity-60"
                >
                  {createProjectMutation.isPending ? "Creating…" : "New project"}
                </button>
              </div>
              {/* Expanded: input + Create — width animates from 0 when expanded */}
              <div
                className="flex shrink-0 items-center overflow-hidden transition-[width] duration-300 ease-out"
                style={{ width: isNewProjectExpanded ? 300 : 0 }}
              >
                <div className="flex w-[300px] items-center">
                  <input
                    ref={newProjectInputRef}
                    type="text"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    onFocus={() => setNewProjectInputFocused(true)}
                    onBlur={() => setNewProjectInputFocused(false)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        handleNewProject();
                      }
                      if (e.key === "Escape") {
                        setNewProjectInputFocused(false);
                        newProjectInputRef.current?.blur();
                      }
                    }}
                    placeholder="Project name"
                    className="min-w-0 flex-1 bg-transparent px-4 py-2 text-sm text-zinc-900 placeholder:text-zinc-500 outline-none"
                  />
                  <button
                    type="button"
                    onClick={() => handleNewProject()}
                    disabled={createProjectMutation.isPending}
                    className="shrink-0 px-4 py-2 text-sm font-semibold text-zinc-900 disabled:opacity-50"
                  >
                    {createProjectMutation.isPending ? "Creating…" : "Create"}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </header>
        <div className="mx-auto w-full max-w-[1700px] px-6 pb-6 pt-4">
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
          {/* Row 1: visualization | architecture */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <div className="h-[420px]">
              <ThreeScene model={modelForPanel ?? null} sustainabilityScores={sustainabilityScores} />
            </div>
            <ModelPanel session={activeSession} model={modelForPanel} />
          </div>

          {/* Row 2: run overview | issues | sustainability */}
          <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <TrainSessionPanel
              session={activeSession}
              onResume={
                activeSession?.status === "pending"
                  ? (id) =>
                      sessionActionMutation.mutate({
                        path: { session_id: id },
                        body: { action: "resume" },
                      })
                  : undefined
              }
              onStop={
                activeSession?.status === "pending"
                  ? (id) =>
                      sessionActionMutation.mutate({
                        path: { session_id: id },
                        body: { action: "stop" },
                      })
                  : undefined
              }
              actionPending={sessionActionMutation.isPending}
            />
            <SessionIssuesPanel
              session={activeSession}
              health={healthData ?? null}
              healthLoading={isHealthLoading}
            />
            <SustainabilityPanel sessionId={selectedSessionId} />
          </div>

          {/* Row 3: recent steps | project trend */}
          <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
            <TrainStepList
              session={activeSession}
              steps={apiSteps}
              stepsLoading={isStepsLoading}
            />
            <ProjectTrendChart projectId={selectedProjectId} />
          </div>
        </div>
      </div>

      {/* ── Combined bottom panel: Runtime output (left tab) + Agent (right tab) ── */}
      <BottomTerminalPanel
        session={activeSession}
        logs={logsForPanel}
        logsLoading={isLogsLoading}
        isOpen={isConsoleOpen}
        onToggleOpen={() => setIsConsoleOpen((o) => !o)}
        consoleHeight={consoleHeight}
        onDragStart={handleDragStart}
        consoleFollow={consoleFollow}
        onToggleFollow={() => setConsoleFollow((f) => !f)}
        consoleBodyRef={consoleBodyRef}
        onConsoleScroll={(e) => {
          const el = e.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 8;
          if (!atBottom && consoleFollow) setConsoleFollow(false);
          if (atBottom && !consoleFollow) setConsoleFollow(true);
        }}
        onPanelHeightChange={handlePanelHeightChange}
        sessionId={selectedSessionId}
        projectId={selectedProjectId}
      />
    </div>
  );
}
