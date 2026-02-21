import { useEffect, useRef, useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const MAX_RECONNECT_DELAY = 30_000;
const INITIAL_RECONNECT_DELAY = 1_000;

/**
 * Mapping from SSE event type → React Query `_id` strings to invalidate.
 */
const INVALIDATION_MAP: Record<string, string[]> = {
  "session.created": [
    "getTrainSessionsSessionsProjectProjectIdGet",
    "getProjectsProjectsGet",
  ],
  "session.updated": [
    "getTrainSessionSessionsSessionIdGet",
    "getTrainSessionsSessionsProjectProjectIdGet",
    "getProjectsProjectsGet",
  ],
  "session.status_changed": [
    "getTrainSessionSessionsSessionIdGet",
    "getTrainSessionsSessionsProjectProjectIdGet",
    "getTrainSessionStatusSessionsSessionIdStatusGet",
    "getProjectsProjectsGet",
  ],
  "step.registered": [
    "getStepsSessionsSessionIdStepGet",
    "getTrainSessionSessionsSessionIdGet",
  ],
  "log.created": [
    "getSessionLogsSessionsSessionIdLogsGet",
  ],
  "model.registered": [
    "getModelSessionsSessionIdModelGet",
  ],
  "diagnostic.completed": [
    "listSessionDiagnosticRunsDiagnosticsSessionsSessionIdGet",
    "getSessionHealthDiagnosticsSessionsSessionIdHealthGet",
    "getTrainSessionSessionsSessionIdGet",
    "getProjectsProjectsGet",
  ],
};

const EVENT_TYPES = Object.keys(INVALIDATION_MAP);

type UseEventSourceOptions = {
  projectId?: number | null;
  sessionId?: number | null;
  enabled?: boolean;
  onEvent?: (eventType: string, data: Record<string, unknown>) => void;
};

export function useEventSource(options: UseEventSourceOptions = {}) {
  const { projectId, sessionId, enabled = true, onEvent } = options;
  const queryClient = useQueryClient();
  const esRef = useRef<EventSource | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(INITIAL_RECONNECT_DELAY);

  const cleanup = useCallback(() => {
    if (reconnectTimerRef.current != null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (esRef.current != null) {
      esRef.current.close();
      esRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      cleanup();
      return;
    }

    function connect() {
      cleanup();

      const params = new URLSearchParams();
      if (projectId != null) params.set("project_id", String(projectId));
      if (sessionId != null) params.set("session_id", String(sessionId));
      const qs = params.toString();
      const url = `${API_BASE}/events/stream${qs ? `?${qs}` : ""}`;

      const es = new EventSource(url);
      esRef.current = es;

      es.onopen = () => {
        reconnectDelayRef.current = INITIAL_RECONNECT_DELAY;
      };

      es.onerror = () => {
        es.close();
        esRef.current = null;
        const delay = reconnectDelayRef.current;
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY);
        reconnectTimerRef.current = setTimeout(connect, delay);
      };

      for (const eventType of EVENT_TYPES) {
        es.addEventListener(eventType, (e: MessageEvent) => {
          let data: Record<string, unknown> = {};
          try {
            data = JSON.parse(e.data);
          } catch {
            // ignore parse failures
          }

          onEvent?.(eventType, data);

          const queryIds = INVALIDATION_MAP[eventType];
          if (queryIds) {
            for (const id of queryIds) {
              queryClient.invalidateQueries({
                predicate: (query) => {
                  const key = query.queryKey[0];
                  return (
                    typeof key === "object" &&
                    key != null &&
                    "_id" in key &&
                    (key as Record<string, unknown>)._id === id
                  );
                },
              });
            }
          }
        });
      }
    }

    connect();
    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, projectId, sessionId, queryClient]);
}
