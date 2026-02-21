"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { sendClaudePrompt } from "@/actions/agent";
import {
  ModelPanel,
  SAMPLE_LOGS,
  SAMPLE_MODEL,
  SAMPLE_SESSIONS,
  SAMPLE_STEPS,
  SessionList,
  SessionLogList,
  TrainSessionPanel,
  TrainStepList,
} from "./ProjectTrainingPanels";
import ThreeScene from "./ThreeScene";

const STORAGE_KEY = "atlas-projects";
const SELECTED_KEY = "atlas-selected-project-token";

type Project = {
  name: string;
  token: string;
  createdAt: string;
};

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


function generateToken() {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }

  if (typeof globalThis.crypto?.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes)
      .map((value) => value.toString(16).padStart(2, "0"))
      .join("");
  }

  return `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
}

function getNextProjectNumber(projects: Project[]) {
  if (projects.length === 0) return 1;

  let max = 0;
  for (const project of projects) {
    const match = project.name.match(/(\d+)$/);
    if (match) {
      const value = Number(match[1]);
      if (!Number.isNaN(value)) max = Math.max(max, value);
    }
  }

  return max + 1;
}

export default function ProjectsClient({
  fontClassName,
}: ProjectsClientProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedToken, setSelectedToken] = useState<string | null>(null);
  const [isProjectsOpen, setIsProjectsOpen] = useState(true);
  const [claudeReply, setClaudeReply] = useState<string>("");
  const [claudeError, setClaudeError] = useState<string>("");
  const [isClaudeLoading, setIsClaudeLoading] = useState(false);
  const hasLoadedRef = useRef(false);
  const [selectedSessionId, setSelectedSessionId] = useState<number | null>(
    null,
  );

  useEffect(() => {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      hasLoadedRef.current = true;
      return;
    }

    try {
      const parsed = JSON.parse(raw) as Project[];
      if (Array.isArray(parsed)) {
        setProjects(parsed);
      }
    } catch {
      window.localStorage.removeItem(STORAGE_KEY);
    } finally {
      const storedSelected = window.localStorage.getItem(SELECTED_KEY);
      if (storedSelected) {
        setSelectedToken(storedSelected);
      }
      hasLoadedRef.current = true;
    }
  }, []);

  useEffect(() => {
    if (!hasLoadedRef.current) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(projects));
  }, [projects]);

  useEffect(() => {
    if (!hasLoadedRef.current) return;
    if (selectedToken) {
      window.localStorage.setItem(SELECTED_KEY, selectedToken);
    } else {
      window.localStorage.removeItem(SELECTED_KEY);
    }
  }, [selectedToken]);

  useEffect(() => {
    if (!selectedToken) return;
    const exists = projects.some((project) => project.token === selectedToken);
    if (!exists) {
      setSelectedToken(null);
    }
  }, [projects, selectedToken]);

  const totalProjects = useMemo(() => projects.length, [projects.length]);
  const selectedProject = useMemo(
    () => projects.find((project) => project.token === selectedToken) ?? null,
    [projects, selectedToken],
  );
  useEffect(() => {
    if (!selectedProject) {
      setSelectedSessionId(null);
      return;
    }
    const session = SAMPLE_SESSIONS[0];
    setSelectedSessionId(session?.id ?? null);
  }, [selectedProject]);
  const activeSession = useMemo(
    () =>
      selectedProject
        ? SAMPLE_SESSIONS.find((session) => session.id === selectedSessionId) ??
          SAMPLE_SESSIONS[0] ??
          null
        : null,
    [selectedProject, selectedSessionId],
  );

  const handleNewProject = () => {
    const token = generateToken();
    setProjects((prev) => {
      const nextNumber = getNextProjectNumber(prev);
      const project: Project = {
        name: `Project ${nextNumber}`,
        token,
        createdAt: new Date().toISOString(),
      };

      return [project, ...prev];
    });
    setSelectedToken(token);
  };

  const handleSuggestionClick = async (suggestion: string) => {
    if (!selectedProject) return;

    try {
      await fetch("/api/suggestions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          suggestion,
          token: selectedProject.token,
          projectName: selectedProject.name,
        }),
      });
    } catch (error) {
      console.error("Failed to send suggestion", error);
    }
  };

  const handleClaudeTest = async () => {
    setClaudeError("");
    setClaudeReply("");
    setIsClaudeLoading(true);

    try {
      const result = await sendClaudePrompt(
        "What should I search for to find the latest developments in renewable energy?",
      );
      setClaudeReply(result.text || "No response text returned.");
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Claude request failed";
      setClaudeError(message);
    } finally {
      setIsClaudeLoading(false);
    }
  };

  return (
    <div className={`${fontClassName} min-h-screen bg-zinc-900 text-zinc-100`}>
      <div className="relative isolate overflow-hidden">
        <header className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-4 px-6 pt-10">
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
              className="rounded-full bg-white px-5 py-2 text-sm font-semibold text-zinc-900 shadow-lg shadow-white/10"
            >
              New project
            </button>
          </div>
        </header>
        <div className="mx-auto w-full max-w-6xl px-6 pb-12 pt-10">
          <div className="grid gap-6">
            <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                    Local storage
                  </p>
                  <h2 className="text-lg font-semibold">Your projects</h2>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-zinc-500">
                    {totalProjects} total
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
                projects.length === 0 ? (
                  <div className="mt-4 rounded-2xl border border-dashed border-zinc-800 bg-zinc-900/40 px-4 py-5 text-sm text-zinc-400">
                    No projects yet. Click "New project" to generate one.
                  </div>
                ) : (
                  <div
                    className={`mt-4 grid gap-3 ${
                      projects.length > 3
                        ? "dark-scrollbar max-h-80 overflow-y-auto pr-2"
                        : ""
                    }`}
                  >
                    {projects.map((project) => (
                      <button
                        type="button"
                        key={project.token}
                        onClick={() => setSelectedToken(project.token)}
                        aria-pressed={selectedToken === project.token}
                        className={`flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-left transition ${
                          selectedToken === project.token
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
                            {new Date(project.createdAt).toLocaleString()}
                          </p>
                        </div>
                        <div className="flex items-center gap-2 text-xs font-medium">
                          {selectedToken === project.token ? (
                            <span className="rounded-full bg-white/15 px-2 py-1 text-[10px] uppercase tracking-[0.2em] text-white/80">
                              Selected
                            </span>
                          ) : null}
                          <span className="rounded-full bg-zinc-950 px-3 py-1 text-zinc-300">
                            <span className="mr-2 text-zinc-500">Token</span>
                            <span className="font-mono tracking-tight">
                              {project.token}
                            </span>
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                )
              ) : (
                <p className="mt-4 text-xs text-zinc-500">
                  Project list hidden.
                </p>
              )}
            </section>

            <ThreeScene />
            <PipelineSuggestions
              selectedProject={selectedProject}
              onSuggestionClick={handleSuggestionClick}
            />
            <SessionList
              selectedProject={selectedProject}
              sessions={SAMPLE_SESSIONS}
              selectedSessionId={selectedSessionId}
              onSelectSession={setSelectedSessionId}
            />
            <TrainSessionPanel session={activeSession} />
            <ModelPanel session={activeSession} model={SAMPLE_MODEL} />
            <TrainStepList session={activeSession} steps={SAMPLE_STEPS} />
            <SessionLogList session={activeSession} logs={SAMPLE_LOGS} />
            <section className="rounded-3xl border border-zinc-800 bg-zinc-950/60 p-6 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-zinc-500">
                    Claude test
                  </p>
                  <h2 className="text-lg font-semibold">
                    Test server action
                  </h2>
                </div>
                <button
                  type="button"
                  onClick={handleClaudeTest}
                  disabled={isClaudeLoading}
                  className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                    isClaudeLoading
                      ? "cursor-not-allowed bg-zinc-800 text-zinc-400"
                      : "bg-white text-zinc-900 hover:bg-zinc-100"
                  }`}
                >
                  {isClaudeLoading ? "Testing..." : "Test Claude"}
                </button>
              </div>
              <div className="mt-4 rounded-2xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 text-sm text-zinc-200">
                {claudeReply || "No response yet."}
              </div>
              {claudeError ? (
                <p className="mt-3 text-xs text-red-300">{claudeError}</p>
              ) : null}
            </section>
          </div>
        </div>
      </div>
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
