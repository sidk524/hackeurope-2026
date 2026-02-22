"use client";

import { useCallback, useRef, useState } from "react";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type AgentMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  model?: string;
  timestamp: number;
};

export type BeliefState = {
  primary_issue: string;
  confidence: number;
  revision_count: number;
  severity: "healthy" | "watch" | "warning" | "critical";
  sustainability_grade: "A" | "B" | "C" | "D" | "F";
  key_observations: string[];
  recommended_actions: string[];
};

type UseAgentChatOptions = {
  sessionId: number | null;
  projectId: number | null;
};

export function useAgentChat({ sessionId, projectId }: UseAgentChatOptions) {
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [beliefState, setBeliefState] = useState<BeliefState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      setError(null);

      const userMsg: AgentMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: text.trim(),
        timestamp: Date.now(),
      };

      // Build the message history for the API
      const allMessages = [...messages, userMsg];
      setMessages(allMessages);

      setIsLoading(true);

      // Create placeholder for assistant response
      const assistantId = `assistant-${Date.now()}`;
      const assistantMsg: AgentMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await fetch(`${API_BASE}/agent/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            project_id: projectId,
            messages: allMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            belief_state: beliefState,
          }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errText = await response.text();
          throw new Error(errText || `HTTP ${response.status}`);
        }

        // Parse SSE stream
        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE events from buffer
          const parts = buffer.split("\n\n");
          // Keep the last incomplete part in the buffer
          buffer = parts.pop() ?? "";

          for (const part of parts) {
            const lines = part.split("\n");
            let eventType = "";
            let data = "";

            for (const line of lines) {
              if (line.startsWith("event: ")) {
                eventType = line.slice(7);
              } else if (line.startsWith("data: ")) {
                data += line.slice(6);
              }
            }

            if (!data) continue;

            try {
              const parsed = JSON.parse(data);

              if (eventType === "message") {
                // Update the assistant message with full content
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantId
                      ? {
                          ...m,
                          content: parsed.content ?? "",
                          model: parsed.model,
                        }
                      : m
                  )
                );
              } else if (eventType === "belief") {
                setBeliefState(parsed as BeliefState);
              } else if (eventType === "error") {
                setError(parsed.detail ?? "Agent error");
              }
              // "done" event — nothing extra to do
            } catch {
              // ignore malformed JSON
            }
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        const errorMsg =
          err instanceof Error ? err.message : "Failed to reach agent";
        setError(errorMsg);
        // Remove the empty assistant placeholder on error
        setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      } finally {
        setIsLoading(false);
        abortRef.current = null;
      }
    },
    [messages, beliefState, sessionId, projectId, isLoading]
  );

  const clearHistory = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setBeliefState(null);
    setError(null);
    setIsLoading(false);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setIsLoading(false);
  }, []);

  return {
    messages,
    beliefState,
    isLoading,
    error,
    sendMessage,
    clearHistory,
    abort,
  };
}
