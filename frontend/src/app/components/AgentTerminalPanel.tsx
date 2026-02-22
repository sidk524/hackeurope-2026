"use client";

import {
    useAgentChat,
    type AgentMessage,
    type BeliefState,
} from "@/lib/use-agent-chat";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ── Severity / grade colors ──────────────────────────────────────────────────

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

// ── Quick‐action presets ─────────────────────────────────────────────────────

const QUICK_ACTIONS = [
  { label: "health", prompt: "Give me a health summary of this training run. What are the top issues?" },
  { label: "sustainability", prompt: "Generate a full sustainability report. How green is this training run?" },
  { label: "architecture", prompt: "Analyze the model architecture. Are there any inefficiencies or suggestions?" },
  { label: "diagnose", prompt: "Run a fresh diagnostic analysis and tell me what you find." },
] as const;

// ── Belief state inline ──────────────────────────────────────────────────────

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
        <span className={`font-bold ${GRADE_COLORS[grade] ?? "text-zinc-400"}`}>
          {grade}
        </span>
        <span className="text-zinc-600">|</span>
        <span className="text-zinc-500">
          confidence {Math.round(belief.confidence * 100)}%
        </span>
        {belief.revision_count > 0 && (
          <>
            <span className="text-zinc-600">|</span>
            <span className="text-amber-400/80">
              {belief.revision_count} rev
            </span>
          </>
        )}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`ml-auto text-zinc-600 transition ${expanded ? "rotate-180" : ""}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-2 space-y-1.5 border-t border-zinc-800/60 pt-2 font-mono text-[11px]">
          <p className="text-zinc-400">
            <span className="text-zinc-600">issue:</span> {belief.primary_issue}
          </p>
          {belief.key_observations?.length > 0 && (
            <div className="text-zinc-400">
              <span className="text-zinc-600">observations:</span>
              {belief.key_observations.map((o, i) => (
                <p key={i} className="ml-3 text-zinc-500">- {o}</p>
              ))}
            </div>
          )}
          {belief.recommended_actions?.length > 0 && (
            <div className="text-zinc-400">
              <span className="text-zinc-600">actions:</span>
              {belief.recommended_actions.map((a, i) => (
                <p key={i} className="ml-3 text-emerald-400/70">→ {a}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Terminal message entry ───────────────────────────────────────────────────

function TerminalEntry({ msg }: { msg: AgentMessage }) {
  const isUser = msg.role === "user";

  // Strip <belief> blocks from display
  const displayContent = msg.content
    .replace(/<belief>[\s\S]*?```json[\s\S]*?```/g, "")
    .replace(/<belief>[\s\S]*?\{[\s\S]*?\}/g, "")
    .trim();

  if (isUser) {
    return (
      <div className="group flex items-start gap-2 py-1">
        <span className="shrink-0 select-none font-mono text-xs font-bold text-indigo-400">
          $
        </span>
        <span className="font-mono text-xs text-zinc-100">{msg.content}</span>
      </div>
    );
  }

  return (
    <div className="py-1.5 pl-4 border-l-2 border-zinc-800/60">
      <div className="mb-1 flex items-center gap-2">
        <span className="flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-[8px] font-bold text-white">
          A
        </span>
        <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-indigo-400/80">
          atlas
        </span>
        {msg.model && (
          <span className="font-mono text-[9px] text-zinc-700">{msg.model}</span>
        )}
      </div>
      <div className="prose prose-invert prose-sm max-w-none break-words font-mono text-xs leading-relaxed prose-p:my-1 prose-headings:my-1.5 prose-headings:text-zinc-200 prose-headings:text-xs prose-li:my-0 prose-ul:my-0.5 prose-ol:my-0.5 prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-indigo-300 prose-code:text-[11px] prose-code:before:content-none prose-code:after:content-none prose-pre:my-1.5 prose-pre:rounded-lg prose-pre:bg-zinc-900 prose-pre:border prose-pre:border-zinc-800 prose-a:text-indigo-400 prose-strong:text-zinc-100 prose-blockquote:border-indigo-800/50 prose-blockquote:text-zinc-500 prose-table:text-[10px] prose-th:text-zinc-300 prose-td:text-zinc-400">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {displayContent}
        </ReactMarkdown>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

type AgentTerminalPanelProps = {
  sessionId: number | null;
  projectId: number | null;
};

export default function AgentTerminalPanel({
  sessionId,
  projectId,
}: AgentTerminalPanelProps) {
  const [input, setInput] = useState("");
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { messages, beliefState, isLoading, error, sendMessage, clearHistory } =
    useAgentChat({ sessionId, projectId });

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  // Focus input when panel gains attention
  const focusInput = useCallback(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      const trimmed = input.trim();
      if (!trimmed) return;

      // Handle built-in commands
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
    (prompt: string) => {
      sendMessage(prompt);
      focusInput();
    },
    [sendMessage, focusInput]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
      // Command history navigation
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
          if (next < 0) {
            setInput("");
            return -1;
          }
          if (commandHistory[next]) setInput(commandHistory[next]);
          return next;
        });
      }
    },
    [handleSubmit, commandHistory]
  );

  const noSession = sessionId == null;

  return (
    <section
      className="flex flex-col rounded-3xl border border-zinc-800 bg-zinc-950/80 shadow-lg overflow-hidden"
      onClick={focusInput}
    >
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/60 px-5 py-3">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <span className="h-3 w-3 rounded-full bg-red-500/80" />
            <span className="h-3 w-3 rounded-full bg-amber-500/80" />
            <span className="h-3 w-3 rounded-full bg-emerald-500/80" />
          </div>
          <div className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded bg-indigo-600 text-[9px] font-bold text-white">
              A
            </span>
            <span className="font-mono text-xs font-semibold text-zinc-300">
              Atlas
            </span>
            <span className="font-mono text-[10px] text-zinc-600">
              — training agent
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {sessionId != null && (
            <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-500">
              session {sessionId}
            </span>
          )}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              clearHistory();
            }}
            className="rounded-lg p-1.5 text-zinc-600 transition hover:bg-zinc-800 hover:text-zinc-400"
            aria-label="Clear terminal"
            title="clear"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-2 14H7L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" /></svg>
          </button>
        </div>
      </div>

      {/* Belief state bar */}
      {beliefState && <BeliefStatusBar belief={beliefState} />}

      {/* Terminal output area */}
      <div
        ref={scrollRef}
        className="flex-1 min-h-[280px] max-h-[520px] overflow-y-auto px-4 py-3 font-mono text-xs"
      >
        {messages.length === 0 && !isLoading ? (
          <div className="space-y-3 py-4">
            {/* Welcome */}
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

            {/* Quick commands */}
            {!noSession && (
              <div className="flex flex-wrap gap-2 pt-1">
                {QUICK_ACTIONS.map((qa) => (
                  <button
                    key={qa.label}
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleQuickAction(qa.prompt);
                    }}
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
            {messages.map((msg) => (
              <TerminalEntry key={msg.id} msg={msg} />
            ))}
            {isLoading && (
              <div className="flex items-center gap-2 py-1.5 pl-4">
                <span className="flex h-4 w-4 items-center justify-center rounded-full bg-indigo-600 text-[8px] font-bold text-white">
                  A
                </span>
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
      {error && (
        <div className="border-t border-red-900/40 bg-red-950/20 px-4 py-1.5 font-mono text-[11px] text-red-400">
          <span className="text-red-600">err:</span> {error}
        </div>
      )}

      {/* Input prompt */}
      <form
        onSubmit={handleSubmit}
        className="flex items-center gap-2 border-t border-zinc-800 bg-zinc-900/40 px-4 py-2.5"
      >
        <span className="shrink-0 select-none font-mono text-xs font-bold text-indigo-400">
          $
        </span>
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            noSession ? "select a session first…" : "ask atlas…"
          }
          disabled={noSession || isLoading}
          autoComplete="off"
          spellCheck={false}
          className="flex-1 bg-transparent font-mono text-xs text-zinc-100 placeholder-zinc-600 outline-none caret-indigo-400 disabled:cursor-not-allowed disabled:opacity-40"
        />
        {isLoading && (
          <span className="shrink-0 font-mono text-[10px] text-zinc-600 animate-pulse">
            thinking…
          </span>
        )}
      </form>
    </section>
  );
}
