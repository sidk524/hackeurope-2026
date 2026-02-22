"use client";

import {
    useAgentChat,
    type AgentMessage,
    type BeliefState,
} from "@/lib/use-agent-chat";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ── Severity badge helpers ───────────────────────────────────────────────────

const SEVERITY_COLORS: Record<string, string> = {
  healthy: "bg-emerald-900/50 text-emerald-300 border-emerald-700",
  watch: "bg-blue-900/50 text-blue-300 border-blue-700",
  warning: "bg-amber-900/50 text-amber-300 border-amber-700",
  critical: "bg-red-900/50 text-red-300 border-red-700",
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
  { label: "Health summary", prompt: "Give me a health summary of this training run. What are the top issues?" },
  { label: "Sustainability", prompt: "Generate a full sustainability report. How green is this training run?" },
  { label: "Architecture", prompt: "Analyze the model architecture. Are there any inefficiencies or suggestions?" },
  { label: "Run diagnostics", prompt: "Run a fresh diagnostic analysis and tell me what you find." },
] as const;

// ── Belief state display ─────────────────────────────────────────────────────

function BeliefStateCard({ belief }: { belief: BeliefState }) {
  const [expanded, setExpanded] = useState(false);
  const sev = belief.severity ?? "watch";
  const grade = belief.sustainability_grade ?? "-";

  return (
    <div className="rounded-2xl border border-zinc-700/60 bg-zinc-900/80 px-4 py-3">
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        className="flex w-full items-center justify-between gap-2 text-left"
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">
            Belief
          </span>
          <span
            className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase ${SEVERITY_COLORS[sev] ?? SEVERITY_COLORS.watch}`}
          >
            {sev}
          </span>
          <span
            className={`text-sm font-bold ${GRADE_COLORS[grade] ?? "text-zinc-400"}`}
            title="Sustainability grade"
          >
            🌱 {grade}
          </span>
          {belief.revision_count > 0 && (
            <span className="rounded-full bg-amber-900/50 px-2 py-0.5 text-[10px] text-amber-300">
              {belief.revision_count} revision{belief.revision_count > 1 ? "s" : ""}
            </span>
          )}
        </div>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`text-zinc-500 transition ${expanded ? "rotate-180" : ""}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {expanded && (
        <div className="mt-3 space-y-2 text-xs text-zinc-300">
          <p>
            <span className="text-zinc-500">Issue:</span>{" "}
            {belief.primary_issue}
          </p>
          <p>
            <span className="text-zinc-500">Confidence:</span>{" "}
            {Math.round(belief.confidence * 100)}%
          </p>
          {belief.key_observations?.length > 0 && (
            <div>
              <span className="text-zinc-500">Observations:</span>
              <ul className="ml-4 mt-1 list-disc space-y-0.5">
                {belief.key_observations.map((o, i) => (
                  <li key={i}>{o}</li>
                ))}
              </ul>
            </div>
          )}
          {belief.recommended_actions?.length > 0 && (
            <div>
              <span className="text-zinc-500">Actions:</span>
              <ul className="ml-4 mt-1 list-disc space-y-0.5">
                {belief.recommended_actions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: AgentMessage }) {
  const isUser = msg.role === "user";

  // Strip <belief> blocks from display
  const displayContent = msg.content
    .replace(/<belief>[\s\S]*?```json[\s\S]*?```/g, "")
    .replace(/<belief>[\s\S]*?\{[\s\S]*?\}/g, "")
    .trim();

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-white/10 text-zinc-100"
            : "border border-zinc-800 bg-zinc-950/80 text-zinc-200"
        }`}
      >
        {!isUser && (
          <div className="mb-1.5 flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-600 text-[9px] font-bold text-white">
              A
            </span>
            <span className="text-[10px] font-semibold uppercase tracking-widest text-indigo-400">
              Atlas
            </span>
            {msg.model && (
              <span className="text-[9px] text-zinc-600">{msg.model}</span>
            )}
          </div>
        )}
        <div className="prose prose-invert prose-sm max-w-none break-words prose-p:my-1.5 prose-headings:my-2 prose-headings:text-zinc-100 prose-li:my-0.5 prose-ul:my-1 prose-ol:my-1 prose-code:rounded prose-code:bg-zinc-800 prose-code:px-1 prose-code:py-0.5 prose-code:text-indigo-300 prose-code:before:content-none prose-code:after:content-none prose-pre:my-2 prose-pre:rounded-lg prose-pre:bg-zinc-800/80 prose-pre:border prose-pre:border-zinc-700/50 prose-a:text-indigo-400 prose-strong:text-zinc-100 prose-blockquote:border-indigo-600/50 prose-blockquote:text-zinc-400 prose-table:text-xs prose-th:text-zinc-300 prose-td:text-zinc-400">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {displayContent}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

// ── Main component ───────────────────────────────────────────────────────────

type AgentChatPanelProps = {
  sessionId: number | null;
  projectId: number | null;
};

export default function AgentChatPanel({
  sessionId,
  projectId,
}: AgentChatPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { messages, beliefState, isLoading, error, sendMessage, clearHistory } =
    useAgentChat({ sessionId, projectId });

  // Auto-scroll on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = useCallback(
    (e?: React.FormEvent) => {
      e?.preventDefault();
      if (!input.trim()) return;
      sendMessage(input);
      setInput("");
    },
    [input, sendMessage]
  );

  const handleQuickAction = useCallback(
    (prompt: string) => {
      sendMessage(prompt);
    },
    [sendMessage]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  // Toggle button (always visible)
  const toggleButton = (
    <button
      type="button"
      onClick={() => setIsOpen((p) => !p)}
      className={`fixed bottom-6 left-6 z-50 flex h-14 w-14 items-center justify-center rounded-full shadow-xl transition-all ${
        isOpen
          ? "bg-zinc-700 text-white hover:bg-zinc-600"
          : "bg-indigo-600 text-white hover:bg-indigo-500 shadow-indigo-900/40"
      }`}
      aria-label={isOpen ? "Close Atlas agent" : "Open Atlas agent"}
    >
      {isOpen ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></svg>
      )}
    </button>
  );

  if (!isOpen) return toggleButton;

  return (
    <>
      {toggleButton}

      {/* Chat panel */}
      <div className="fixed bottom-24 left-6 z-50 flex h-[600px] w-[420px] flex-col overflow-hidden rounded-3xl border border-zinc-700/70 bg-zinc-900 shadow-2xl shadow-black/50">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-3">
          <div className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-indigo-600 text-xs font-bold text-white">
              A
            </span>
            <div>
              <h3 className="text-sm font-semibold text-zinc-100">Atlas</h3>
              <p className="text-[10px] text-zinc-500">
                Adaptive training agent
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            {sessionId != null && (
              <span className="rounded-full bg-zinc-800 px-2 py-0.5 text-[10px] text-zinc-400">
                Session {sessionId}
              </span>
            )}
            <button
              type="button"
              onClick={clearHistory}
              className="rounded-lg p-1.5 text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-300"
              aria-label="Clear chat"
              title="Clear chat"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-2 14H7L5 6" /><path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4h6v2" /></svg>
            </button>
          </div>
        </div>

        {/* Belief state bar */}
        {beliefState && (
          <div className="border-b border-zinc-800 px-4 py-2">
            <BeliefStateCard belief={beliefState} />
          </div>
        )}

        {/* Messages */}
        <div
          ref={scrollRef}
          className="flex-1 space-y-3 overflow-y-auto px-4 py-4"
        >
          {messages.length === 0 && !isLoading ? (
            <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-600/20 text-indigo-400">
                <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
              </div>
              <div>
                <p className="text-sm font-medium text-zinc-300">
                  Ask Atlas about your training run
                </p>
                <p className="mt-1 text-xs text-zinc-500">
                  Health diagnostics, sustainability analysis, architecture advice
                </p>
              </div>

              {/* Quick actions */}
              <div className="mt-2 grid w-full gap-2">
                {QUICK_ACTIONS.map((qa) => (
                  <button
                    key={qa.label}
                    type="button"
                    onClick={() => handleQuickAction(qa.prompt)}
                    disabled={sessionId == null}
                    className="rounded-xl border border-zinc-800 bg-zinc-950/60 px-3 py-2.5 text-left text-xs text-zinc-300 transition hover:border-zinc-600 hover:bg-zinc-900 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {qa.label}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <MessageBubble key={msg.id} msg={msg} />
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-2xl border border-zinc-800 bg-zinc-950/80 px-4 py-3">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-600 text-[9px] font-bold text-white">
                      A
                    </span>
                    <div className="flex gap-1">
                      <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" style={{ animationDelay: "0ms" }} />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" style={{ animationDelay: "150ms" }} />
                      <span className="h-2 w-2 animate-bounce rounded-full bg-zinc-500" style={{ animationDelay: "300ms" }} />
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Error banner */}
        {error && (
          <div className="border-t border-red-900/50 bg-red-950/30 px-4 py-2 text-xs text-red-300">
            {error}
          </div>
        )}

        {/* Input */}
        <form
          onSubmit={handleSubmit}
          className="flex items-end gap-2 border-t border-zinc-800 px-4 py-3"
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              sessionId == null
                ? "Select a session first…"
                : "Ask Atlas…"
            }
            disabled={sessionId == null || isLoading}
            rows={1}
            className="flex-1 resize-none rounded-xl border border-zinc-700 bg-zinc-800 px-3 py-2 text-sm text-zinc-100 placeholder-zinc-500 outline-none transition focus:border-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={!input.trim() || sessionId == null || isLoading}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
            aria-label="Send"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13" /><polygon points="22 2 15 22 11 13 2 9 22 2" /></svg>
          </button>
        </form>
      </div>
    </>
  );
}
