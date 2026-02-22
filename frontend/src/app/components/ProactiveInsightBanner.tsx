"use client";

import { useCallback, useEffect, useState } from "react";

export type InsightItem = {
  id: string;
  severity: string;
  title: string;
  body: string;
  is_revision: boolean;
  timestamp: number;
  dismissed: boolean;
};

const SEVERITY_STYLES: Record<string, { border: string; bg: string; icon: string; text: string }> = {
  critical: {
    border: "border-red-700/60",
    bg: "bg-red-950/40",
    icon: "🔴",
    text: "text-red-300",
  },
  warning: {
    border: "border-amber-700/60",
    bg: "bg-amber-950/40",
    icon: "🟡",
    text: "text-amber-300",
  },
  watch: {
    border: "border-blue-700/60",
    bg: "bg-blue-950/40",
    icon: "🔵",
    text: "text-blue-300",
  },
  healthy: {
    border: "border-emerald-700/60",
    bg: "bg-emerald-950/40",
    icon: "🟢",
    text: "text-emerald-300",
  },
};

const DEFAULT_STYLE = SEVERITY_STYLES.watch;

type ProactiveInsightBannerProps = {
  insights: InsightItem[];
  onDismiss: (id: string) => void;
  onAskAtlas: (insight: InsightItem) => void;
};

export default function ProactiveInsightBanner({
  insights,
  onDismiss,
  onAskAtlas,
}: ProactiveInsightBannerProps) {
  const visible = insights.filter((i) => !i.dismissed);

  if (visible.length === 0) return null;

  return (
    <div className="space-y-2">
      {visible.map((insight) => {
        const style = SEVERITY_STYLES[insight.severity] ?? DEFAULT_STYLE;

        return (
          <div
            key={insight.id}
            className={`animate-in slide-in-from-top-2 rounded-2xl border ${style.border} ${style.bg} px-5 py-4 shadow-lg transition-all`}
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span>{style.icon}</span>
                  <span className={`text-sm font-semibold ${style.text}`}>
                    {insight.title}
                  </span>
                  {insight.is_revision && (
                    <span className="rounded-full bg-amber-800/50 px-2 py-0.5 text-[10px] font-semibold text-amber-200">
                      Updated Assessment
                    </span>
                  )}
                </div>
                <p className="mt-1.5 text-xs leading-relaxed text-zinc-300">
                  {insight.body.length > 300
                    ? insight.body.slice(0, 300) + "…"
                    : insight.body}
                </p>
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => onAskAtlas(insight)}
                    className="rounded-lg bg-indigo-600/30 px-3 py-1 text-[11px] font-medium text-indigo-300 transition hover:bg-indigo-600/50"
                  >
                    Ask Atlas to explain
                  </button>
                  <span className="text-[10px] text-zinc-600">
                    {new Date(insight.timestamp).toLocaleTimeString()}
                  </span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => onDismiss(insight.id)}
                className="shrink-0 rounded-lg p-1 text-zinc-500 transition hover:bg-zinc-800 hover:text-zinc-300"
                aria-label="Dismiss"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/**
 * Hook to manage proactive insights state. Receives new insights
 * from SSE events and auto-dismisses after 60s.
 */
export function useProactiveInsights() {
  const [insights, setInsights] = useState<InsightItem[]>([]);

  const addInsight = useCallback(
    (data: { severity?: string; title?: string; body?: string; is_revision?: boolean }) => {
      const item: InsightItem = {
        id: `insight-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        severity: data.severity ?? "watch",
        title: data.title ?? "Training Update",
        body: data.body ?? "",
        is_revision: data.is_revision ?? false,
        timestamp: Date.now(),
        dismissed: false,
      };
      setInsights((prev) => [item, ...prev].slice(0, 10)); // keep max 10
    },
    []
  );

  const dismiss = useCallback((id: string) => {
    setInsights((prev) =>
      prev.map((i) => (i.id === id ? { ...i, dismissed: true } : i))
    );
  }, []);

  // Auto-dismiss after 60s
  useEffect(() => {
    const interval = setInterval(() => {
      const cutoff = Date.now() - 60_000;
      setInsights((prev) =>
        prev.map((i) =>
          !i.dismissed && i.timestamp < cutoff ? { ...i, dismissed: true } : i
        )
      );
    }, 10_000);
    return () => clearInterval(interval);
  }, []);

  return { insights, addInsight, dismiss };
}
