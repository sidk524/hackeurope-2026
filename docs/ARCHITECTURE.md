# Architecture — Deep Technical Reference

> **Project:** Atlas — ML Training Observatory & Green-AI Advisor  
> **Event:** HackEurope 2026  
> **Last updated:** 2026-02-22

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture Diagram](#2-high-level-architecture-diagram)
3. [Component Inventory](#3-component-inventory)
4. [Data Flow — End to End](#4-data-flow--end-to-end)
5. [Backend (FastAPI)](#5-backend-fastapi)
   - 5.1 [Application Entrypoint](#51-application-entrypoint)
   - 5.2 [Database Layer](#52-database-layer)
   - 5.3 [ORM Models](#53-orm-models)
   - 5.4 [Routers (API Surface)](#54-routers-api-surface)
   - 5.5 [Diagnostics Engine](#55-diagnostics-engine)
   - 5.6 [Event Bus & SSE](#56-event-bus--sse)
   - 5.7 [Agent System (Atlas)](#57-agent-system-atlas)
   - 5.8 [MCP Server](#58-mcp-server)
   - 5.9 [LLM Proxy](#59-llm-proxy)
6. [Frontend (Next.js)](#6-frontend-nextjs)
   - 6.1 [App Structure](#61-app-structure)
   - 6.2 [State Management](#62-state-management)
   - 6.3 [Component Map](#63-component-map)
   - 6.4 [Real-Time Updates](#64-real-time-updates)
   - 6.5 [3D Visualization](#65-3d-visualization)
7. [Neural Network Observer](#7-neural-network-observer)
   - 7.1 [Observer Class](#71-observer-class)
   - 7.2 [ObserverConfig](#72-observerconfig)
   - 7.3 [Telemetry Channels](#73-telemetry-channels)
   - 7.4 [Report Schema (Pydantic)](#74-report-schema-pydantic)
   - 7.5 [Backend Sync Protocol](#75-backend-sync-protocol)
8. [Standalone Visualization (Vite + TensorSpace)](#8-standalone-visualization-vite--tensorspace)
9. [Cross-Cutting Concerns](#9-cross-cutting-concerns)
10. [Design Decisions & Rationale](#10-design-decisions--rationale)

---

## 1. System Overview

Atlas is a **real-time ML training observatory** that watches PyTorch models train, collects rich telemetry (loss, throughput, memory, profiler, layer-health, carbon emissions), runs 30+ automated diagnostic checks, and presents findings through:

- A **Next.js dashboard** with live-updating charts, health scores, and sustainability grades
- An **AI agent ("Atlas")** that uses tool-calling to query training data and provide contextual advice
- A **3D architecture visualizer** using TensorSpace for interactive neural network exploration
- A **Green-AI sustainability engine** that tracks CO₂ emissions, identifies wasted compute, and grades training efficiency

The system addresses two HackEurope 2026 challenge tracks:

| Challenge | How Atlas Addresses It |
|-----------|----------------------|
| **incident.io — Adaptive Agent** | Atlas maintains a belief state, revises assessments when new data contradicts prior analysis, and proactively alerts on training anomalies |
| **Green Code Optimizer — Sustainability** | Tracks carbon emissions per-epoch, identifies dead layers / vanishing gradients / wasted epochs, grades training A–F, estimates carbon cost in EUR |

---

## 2. High-Level Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING ENVIRONMENT                               │
│                                                                               │
│   Jupyter Notebook (mini_gpt.ipynb / small_cnn.ipynb / buggy_cnn.ipynb)      │
│     │                                                                         │
│     ├── PyTorch Model (GPTLanguageModel / SmallCNN)                          │
│     │                                                                         │
│     └── Observer (observer.py)                                                │
│           ├── Session metadata, hyperparameters, model architecture           │
│           ├── Per-step: loss, throughput, memory, profiler, layer health      │
│           ├── Carbon emissions (CodeCarbon)                                    │
│           ├── Console & error log capture                                     │
│           └── HTTP POST ──────────────────────────┐                           │
│                                                    │                          │
└────────────────────────────────────────────────────┼──────────────────────────┘
                                                     │
                                                     ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            BACKEND (FastAPI)                                  │
│                          http://localhost:8000                                 │
│                                                                               │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐   │
│   │  /projects   │  │  /sessions   │  │ /diagnostics │  │    /agent      │   │
│   │  CRUD        │  │  CRUD, Steps │  │  Engine, Run │  │  Chat, Analyze │   │
│   │  projects    │  │  Logs, Model │  │  Health, Fix │  │  Tool-calling  │   │
│   └──────────────┘  └──────────────┘  └──────────────┘  └────────────────┘   │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│   │  /events     │  │    /llm      │  │  MCP Server  │                        │
│   │  SSE stream  │  │  LLM proxy   │  │  (stdio/SSE) │                        │
│   └──────────────┘  └──────────────┘  └──────────────┘                        │
│                                                                               │
│   ┌────────────────────────────────────────────────────────┐                  │
│   │  SQLite Database (app.db)                               │                  │
│   │  Tables: Project, TrainSession, Model, TrainStep,       │                  │
│   │          SessionLog, DiagnosticRun, DiagnosticIssue     │                  │
│   └────────────────────────────────────────────────────────┘                  │
│                                                                               │
│   ┌────────────────────────────────────────────────────────┐                  │
│   │  Diagnostics Engine (diagnostics/engine.py)             │                  │
│   │  30+ heuristic checks: loss, system, profiler,          │                  │
│   │  sustainability, architecture-specific (CNN/Transformer) │                  │
│   └────────────────────────────────────────────────────────┘                  │
│                                                                               │
│   ┌────────────────────────────────────────────────────────┐                  │
│   │  Event Bus (event_bus.py)                               │                  │
│   │  In-memory pub/sub → SSE push to frontend               │                  │
│   └────────────────────────────────────────────────────────┘                  │
└───────────────────────────────────────────────────────────────────────────────┘
                           │                    │
                    SSE stream           REST API calls
                           │                    │
                           ▼                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (Next.js 16)                                │
│                         http://localhost:3000                                  │
│                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │  ProjectsClient.tsx — Main orchestrator                              │    │
│   │  ├── ProjectTrendChart — Cross-session trend (Recharts)              │    │
│   │  ├── StepsDashboard — Per-epoch loss/throughput/memory charts         │    │
│   │  ├── SustainabilityPanel — Green-AI grades, CO₂, waste analysis      │    │
│   │  ├── ThreeScene — 3D TensorSpace neural network visualization        │    │
│   │  ├── AgentTerminalPanel — Chat with Atlas agent                      │    │
│   │  ├── ProactiveInsightBanner — Auto-generated training alerts          │    │
│   │  └── ProjectTrainingPanels — Session list, model info, issues        │    │
│   └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐      │
│   │ React Query       │  │ useEventSource   │  │ useAgentChat         │      │
│   │ (data fetching)   │  │ (SSE listener)   │  │ (agent streaming)    │      │
│   └──────────────────┘  └──────────────────┘  └──────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                    STANDALONE VISUALIZATION (Vite)                             │
│                       http://localhost:5173                                    │
│                                                                               │
│   TensorSpace.js model fed from Observer JSON reports (mock-report.json)      │
│   Renders Conv2d → MaxPool2d → Flatten → Dense → Output as interactive 3D    │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Inventory

| Component | Directory | Language | Framework | Purpose |
|-----------|-----------|----------|-----------|---------|
| Backend API | `backend/` | Python 3.12+ | FastAPI 0.129 | REST API, diagnostics engine, event bus, agent orchestration |
| Frontend Dashboard | `frontend/` | TypeScript | Next.js 16, React 19 | Interactive training dashboard with real-time updates |
| Neural Network Observer | `neural_network/` | Python | PyTorch | Training telemetry collector, report generator |
| Standalone Visualizer | `visualization/` | JavaScript | Vite, TensorSpace, Three.js | 3D neural network architecture visualization |

---

## 4. Data Flow — End to End

### Phase 1: Training & Data Collection

```
Jupyter Notebook
    │
    ├── 1. obs = Observer(project_id=1, config=..., run_name="my-run")
    │       → POST /sessions/project/1  (creates TrainSession in DB)
    │
    ├── 2. obs.register_model(model)
    │       → POST /sessions/{id}/model  (sends architecture + hyperparams)
    │       → Installs forward/backward hooks for layer health tracking
    │
    ├── 3. Training loop: obs.step(step, loss, batch_size, seq_length)
    │       → Accumulates loss, throughput, memory, system metrics per batch
    │       → If profile_step: runs torch.profiler for op-level timings
    │       → Layer health hooks capture activation/gradient stats per batch
    │       → CodeCarbon tracks energy consumption
    │
    ├── 4. obs.flush(val_metrics={"val_loss": ..., "val_acc": ...})
    │       → Computes epoch summary (sustainability, layer health, carbon)
    │       → POST /sessions/{id}/step  (sends full epoch telemetry)
    │       → POST /sessions/{id}/logs  (sends batched console/error logs)
    │       → Polls GET /sessions/{id}/status for pause/stop signals
    │
    └── 5. obs.close()
            → PATCH /sessions/{id}  (sets ended_at, marks completed/failed)
            → Optionally exports to JSON file (observer_reports/)
```

### Phase 2: Diagnostics

```
POST /sessions/{id}/step  (from Observer)
    │
    ├── Persists TrainStep row
    ├── Publishes SSEEvent(step.registered)
    └── Enqueues background task: _run_step_diagnostics()
          │
          ├── Loads all steps + logs + model architecture
          ├── Calls diagnostics.engine.run_diagnostics()
          │     └── Runs 30+ heuristic checks:
          │           • Loss: divergence, explosion, plateau, overfitting, variance
          │           • System: memory growth, CPU bottleneck, slow epochs
          │           • Profiler: hotspots, backward dominance, per-layer timing
          │           • Sustainability: wasted compute, dead layers, vanishing gradients
          │           • Architecture: CNN-specific (missing pooling, FC dominance)
          │           • Carbon: emission spikes, wasted CO₂
          │
          ├── Persists DiagnosticRun + DiagnosticIssues
          ├── Sets session status (running / pending based on severity)
          └── Publishes SSEEvent(diagnostic.completed)
```

### Phase 3: Frontend Display

```
SSE stream (/events/stream)
    │
    ├── Frontend EventSource receives events
    ├── useEventSource hook invalidates React Query caches
    └── Components auto-refetch via React Query:
          • StepsDashboard → latest loss/throughput/memory charts
          • SustainabilityPanel → CO₂, efficiency grade, waste analysis
          • ProjectTrainingPanels → session status, issues
          • ProactiveInsightBanner → agent-generated alerts
```

### Phase 4: Agent Interaction

```
User types in AgentTerminalPanel
    │
    └── POST /agent/chat (SSE streaming)
          │
          ├── Build system prompt (with session context + belief state)
          ├── Send to LLM (Crusoe/Qwen3 → fallback to Anthropic/Claude)
          ├── Tool-calling loop (max 8 iterations):
          │     LLM calls → get_session_health, get_training_steps, etc.
          │     Tool results fed back → LLM refines answer
          │
          ├── Stream response as SSE events:
          │     event: message → assistant text
          │     event: belief  → updated belief state JSON
          │     event: done    → completion signal
          │
          └── Frontend updates:
                • AgentTerminalPanel: renders markdown response
                • BeliefStatusBar: shows severity, grade, confidence
```

---

## 5. Backend (FastAPI)

### 5.1 Application Entrypoint

**File:** `backend/main.py`

The FastAPI application is created and configured here:

- **CORS**: Allows `localhost:3000` (Next.js), `localhost:5173` (Vite), and `127.0.0.1:3000` by default. Configurable via `CORS_ALLOWED_ORIGINS` env var.
- **Routers**: Six routers are mounted at startup:
  - `projects.router` → `/projects`
  - `sessions.router` → `/sessions`
  - `diagnostics.router` → `/diagnostics`
  - `events.router` → `/events`
  - `llm.router` → `/llm`
  - `agent.router` → `/agent`
- **Startup event**: Captures the asyncio event loop for the sync-to-async bridge used by the event bus.

**Run command:** `uvicorn main:fastapi --reload`

### 5.2 Database Layer

**File:** `backend/database.py`

- **Engine**: SQLite (`sqlite:///./app.db`) by default. Configurable via `DATABASE_URL` env var.
- **ORM**: SQLModel (Pydantic + SQLAlchemy hybrid).
- **Session management**: FastAPI dependency injection via `SessionDep = Annotated[Session, Depends(get_session)]`.
- **Migrations**: Alembic (`backend/alembic/`). Three migrations track schema evolution:
  1. `488e1970ad6c` — Initial tables (Project, TrainSession, Model, TrainStep, SessionLog)
  2. `a1b2c3d4e5f6` — Add DiagnosticRun, DiagnosticIssue tables
  3. `e57e1f3af050` — Add sustainability columns (layer_health, sustainability, carbon_emissions, log_counts) to TrainStep

### 5.3 ORM Models

**File:** `backend/models.py`

| Model | Table | Key Fields | Relationships |
|-------|-------|------------|---------------|
| `Project` | `project` | `id`, `name`, `created_at` | → `sessions[]` |
| `TrainSession` | `trainsession` | `id`, `project_id`, `run_id`, `run_name`, `started_at`, `ended_at`, `device`, `cuda_available`, `pytorch_version`, `config` (JSON), `summary` (JSON), `status` (enum) | → `project`, `steps[]`, `session_logs[]`, `models[]`, `diagnostic_runs[]` |
| `Model` | `model` | `id`, `session_id` (unique), `architecture` (JSON), `hyperparameters` (JSON) | → `session` |
| `TrainStep` | `trainstep` | `id`, `session_id`, `step_index`, `timestamp`, `duration_seconds`, `loss` (JSON), `throughput` (JSON), `profiler` (JSON), `memory` (JSON), `system` (JSON), `layer_health` (JSON), `sustainability` (JSON), `carbon_emissions` (JSON), `log_counts` (JSON) | → `session` |
| `SessionLog` | `sessionlog` | `id`, `session_id`, `ts`, `level`, `msg`, `module`, `lineno`, `kind` (enum: console/error) | → `session` |
| `DiagnosticRun` | `diagnosticrun` | `id`, `session_id`, `created_at`, `health_score`, `issue_count`, `arch_type`, `summary_json` (JSON) | → `session`, `issues[]` |
| `DiagnosticIssue` | `diagnosticissue` | `id`, `run_id`, `severity` (enum), `category` (enum), `title`, `description`, `epoch_index`, `layer_id`, `metric_key`, `metric_value` (JSON), `suggestion` | → `run` |

**Enums:**

- `SessionStatus`: running, completed, failed, pending, analyzing, stopped
- `LogKind`: console, error
- `IssueSeverity`: critical, warning, info
- `IssueCategory`: loss, throughput, memory, profiler, logs, system, architecture, sustainability

### 5.4 Routers (API Surface)

#### Projects Router (`/projects`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/` | Create a new project |
| `GET` | `/` | List all projects (with latest session status) |
| `GET` | `/{project_id}` | Get single project |

#### Sessions Router (`/sessions`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/project/{project_id}` | Create a training session |
| `GET` | `/project/{project_id}` | List sessions for a project |
| `GET` | `/{session_id}` | Get session detail |
| `PATCH` | `/{session_id}` | Update session (ended_at, summary, status) |
| `GET` | `/{session_id}/status` | Get session status |
| `POST` | `/{session_id}/action` | Stop/resume a session |
| `POST` | `/{session_id}/model` | Register model architecture |
| `GET` | `/{session_id}/model` | Get model architecture |
| `POST` | `/{session_id}/step` | Register a training step (triggers background diagnostics) |
| `GET` | `/{session_id}/step` | Get all steps for a session |
| `POST` | `/{session_id}/log` | Create a single log entry |
| `POST` | `/{session_id}/logs` | Batch-create log entries |
| `GET` | `/{session_id}/logs` | Get all logs for a session |

**Key behavior:** When a step is registered, the session status transitions to `analyzing`, a background task runs diagnostics, and the session is set to `running` or `pending` based on severity of found issues.

#### Diagnostics Router (`/diagnostics`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions/{session_id}/run` | Trigger a full diagnostic run |
| `GET` | `/sessions/{session_id}` | List all diagnostic runs for a session |
| `GET` | `/runs/{run_id}` | Get full diagnostic run detail |
| `GET` | `/sessions/{session_id}/health` | Get health summary (latest run or on-the-fly) |
| `GET` | `/projects/{project_id}/trend` | Get cross-session improvement trend |
| `POST` | `/issues/{issue_id}/prompt` | Generate a fix prompt from an issue |
| `GET` | `/issues/{issue_id}/prompt` | Retrieve cached fix prompt |

**Response structure for a diagnostic run (`DiagnosticRunOut`):**
- `issues[]` — full list of found issues
- `epoch_trends[]` — issues grouped by epoch
- `session_level_issues[]` — issues with no specific epoch
- `layer_highlights[]` — issues grouped by layer, sorted by severity score
- `sustainability` — `SustainabilityInsight` with carbon/efficiency/waste analysis

#### Events Router (`/events`)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/stream` | SSE endpoint for real-time push notifications |

Query params: `project_id`, `session_id` for server-side filtering.

#### LLM Router (`/llm`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat/completions` | Proxy to Crusoe Cloud LLM (Qwen3-235B) |

Acts as an OpenAI-compatible proxy. Forwards requests to `https://hackeurope.crusoecloud.com/v1/`. Streams responses back.

#### Agent Router (`/agent`)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/chat` | Multi-turn agent chat with tool calling (SSE streaming) |
| `POST` | `/analyze` | Single-shot proactive analysis (returns structured insight) |

### 5.5 Diagnostics Engine

**File:** `backend/diagnostics/engine.py` (~1807 lines)

The engine is the analytical core. It takes epoch data, logs, architecture, and hyperparameters, then runs a battery of heuristic checks.

#### Architecture Detection

`detect_arch_type()` inspects the layer graph node categories:
- `"attention"` in categories → `"transformer"`
- `"recurrent"` → `"rnn"`
- `"convolution"` → `"cnn"`
- Otherwise → `"generic"`

#### Check Categories

**Generic Checks (all architectures):**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_loss_divergence` | warning | 2+ consecutive epochs with rising train_mean |
| `check_loss_explosion` | critical | >100% jump in loss between epochs |
| `check_loss_plateau` | warning | <1% improvement over last 3 epochs |
| `check_overfitting` | warning | Val loss rising while train loss falls |
| `check_high_loss_variance` | warning | train_std/train_mean > 0.5 |
| `check_gradient_instability` | warning | max_loss/mean > 20x (gradient spike proxy) |

**System Checks:**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_throughput_degradation` | warning | Throughput drops >20% from peak |
| `check_memory_growth` | warning | RSS grows >25% from first to last epoch |
| `check_slow_epoch` | info | Epoch >1.5x median duration |
| `check_high_cpu` | info | CPU usage >90% |

**Profiler Checks:**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_profiler_hotspot` | warning | Layer >40% of CPU time in an epoch |
| `check_consistent_profiler_hot` | info | Layer >25% CPU time for 3+ epochs |
| `check_backward_dominance` | info | Backward pass >45% of total CPU time |

**Sustainability Checks (Profiler-Level):**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_diminishing_returns` | warning/info | Marginal loss improvement < 5% of cumulative |
| `check_over_parameterized_layer` | warning | param%/compute% ratio > 10x |
| `check_compute_inefficient_layer` | warning/info | compute%/param% ratio > 10x |
| `check_device_underutilization` | warning | CUDA/CPU time ratio < 0.1 |
| `check_early_stopping_missed` | warning | Training continued past optimal stop |

**Sustainability Checks (Tensor-Level / Layer Health):**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_dead_layers` | critical | Near-zero activation variance + high weight sparsity |
| `check_vanishing_gradients_tensor` | warning | Gradient norm mean < 1e-7 for 2+ epochs |
| `check_depth_gradient_ratio` | warning | Last/first layer gradient ratio > 100x |
| `check_frozen_output_layers` | warning | Activation variance of means < 1e-8 |
| `check_activation_collapse` | warning | Near-zero activation std for 2+ epochs |
| `check_redundant_layers` | info | Activation correlation >0.95 between layer pairs |

**Carbon Footprint Checks:**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_carbon_intensity_spike` | warning | Epoch CO₂ > 10x session average |
| `check_wasted_carbon` | warning/info | CO₂ emitted after optimal stopping point |
| `check_gpu_idle_waste` | info | GPU powered but utilization <20% |
| `check_cpu_dominant_training` | info | CPU energy > 5x GPU energy |

**CNN-Specific Checks:**

| Check Function | Severity | What It Detects |
|---------------|----------|-----------------|
| `check_fc_dominates_params` | warning | FC layer holds >92% of parameters |
| `check_conv_bottleneck` | info | Conv2d holds >60% of parameters |
| `check_missing_pooling` | warning | 3+ consecutive conv layers without pooling |
| `check_large_kernel` | info | Kernel size ≥7 |
| `check_grayscale_channel_explosion` | info | First conv expands 1→32+ channels |

#### Health Score Calculation

```python
score = 100
for issue in issues:
    score -= SEVERITY_WEIGHT[issue.severity]  # critical=20, warning=7, info=2
return max(0, score)
```

### 5.6 Event Bus & SSE

**File:** `backend/event_bus.py`

An in-memory publish/subscribe system:

- **`EventBus`** class manages subscriber queues (asyncio.Queue)
- **`subscribe()`** returns (sub_id, queue) — frontend connects via SSE
- **`publish(event)`** broadcasts to all subscribers (drops oldest if queue full)
- **`publish_from_sync(event)`** bridges synchronous code (like background threads) to the async event loop

**Event types:**

| Event | Emitted When | Frontend Reaction |
|-------|-------------|-------------------|
| `session.created` | New session created | Refetch session list |
| `session.updated` | Session patched | Refetch session detail |
| `session.status_changed` | Status transitions | Refetch session status |
| `step.registered` | New training step | Refetch steps data |
| `log.created` | New log entry | Refetch logs |
| `model.registered` | Model architecture registered | Refetch model |
| `diagnostic.completed` | Diagnostics finished | Refetch health, diagnostics |
| `agent.insight` | Proactive analysis completed | Show insight banner |

### 5.7 Agent System (Atlas)

**Files:** `backend/routers/agent.py`, `backend/agent_system_prompt.py`, `backend/agent_tools.py`

#### System Prompt (`agent_system_prompt.py`)

The Atlas agent has a detailed persona:
- **Identity**: Concise, technical ML diagnostics advisor
- **Adaptive Protocol**: Maintains a belief state JSON that tracks `primary_issue`, `confidence`, `revision_count`, `severity`, `sustainability_grade`, `key_observations`, `recommended_actions`
- **Green-AI Expertise**: Grades training A–F, converts CO₂ to real-world equivalents, estimates cost using EU ETS €50/ton
- **Architecture Advisor**: Interprets layer graphs and parameter distributions

The system prompt is templated with session context (IDs, status, previous belief state).

#### Tool System (`agent_tools.py`)

10 tools available to the agent:

| Tool | Description |
|------|-------------|
| `get_session_detail` | Full session metadata + step stats |
| `get_training_steps` | Per-epoch telemetry (with optional range filter) |
| `get_session_health` | Health score + severity counts + top issues |
| `get_diagnostic_run_detail` | Full issue list with suggestions |
| `run_session_diagnostics` | Trigger a fresh diagnostic analysis |
| `get_model_architecture` | Module tree, layers, hyperparameters |
| `get_session_logs` | Console & error logs |
| `get_project_trend` | Cross-session improvement trend |
| `get_sustainability_report` | Carbon timeline, waste analysis, efficiency |
| `compare_sessions` | Side-by-side comparison of two runs |

Tools are defined as OpenAI function-calling schemas and executed directly against router functions (no HTTP round-trips).

#### LLM Provider Strategy (`agent.py`)

```
Primary: Crusoe Cloud (Qwen3-235B-A22B-Instruct) via OpenAI-compatible API
    │
    ├── Handles standard OpenAI tool_calls (Path A)
    ├── Handles Qwen3 native <tool_call> tags (Path B)
    └── Strips <think> reasoning blocks from output
    
Fallback: Anthropic (Claude Sonnet 4) via native Anthropic API
    └── Converts tool schemas to Anthropic format
```

**Tool-calling loop**: Max 8 iterations. Each iteration:
1. Send messages to LLM with tool schemas
2. If LLM returns tool calls → execute them → feed results back
3. If no tool calls → return final answer

**Belief state extraction**: Regex-parses `<belief>` JSON blocks from the agent's response text.

### 5.8 MCP Server

**File:** `backend/mcp_server.py`

A Model Context Protocol server (via `fastmcp`) that exposes the same diagnostic tools for use with Claude Desktop or other MCP-compatible clients.

**Transport modes:**
- `stdio` (default) — for Claude Desktop
- `--transport sse` — SSE on port 8100

**Tools exposed:** `list_sessions`, `get_session_detail`, `get_training_steps`, `list_diagnostic_runs`, `get_diagnostic_run_detail`, `get_session_health`, `run_session_diagnostics`, `get_project_trend`, `get_session_logs`, `get_model_architecture`

### 5.9 LLM Proxy

**File:** `backend/routers/llm.py`

A transparent proxy to Crusoe Cloud's OpenAI-compatible API:
- Endpoint: `POST /llm/chat/completions`
- Upstream: `https://hackeurope.crusoecloud.com/v1/`
- Model: `NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4`
- Requires `CRUSOE_API_KEY` environment variable
- Supports streaming and non-streaming responses

---

## 6. Frontend (Next.js)

### 6.1 App Structure

```
frontend/src/
├── app/
│   ├── layout.tsx          # Root layout (IBM Plex Mono font, Providers wrapper)
│   ├── page.tsx            # Home page → renders ProjectsClient
│   ├── providers.tsx       # React Query provider
│   ├── globals.css         # Tailwind v4 + CSS variables
│   └── components/         # Page-scoped components
│       ├── ProjectsClient.tsx           # Main orchestrator (~920 lines)
│       ├── ProjectTrainingPanels.tsx    # Session panels (~1489 lines)
│       ├── StepsDashboard.tsx           # Training charts (~1408 lines)
│       ├── SustainabilityPanel.tsx      # Green-AI panel (~307 lines)
│       ├── ThreeScene.tsx              # 3D visualization (~598 lines)
│       ├── AgentTerminalPanel.tsx       # Agent chat UI (~403 lines)
│       ├── ProactiveInsightBanner.tsx   # Alert banners (~160 lines)
│       └── ProjectTrendChart.tsx        # Cross-session trends (~333 lines)
├── components/ui/
│   └── button.tsx           # shadcn button primitive
├── lib/
│   ├── utils.ts             # clsx + tailwind-merge helper
│   ├── use-agent-chat.ts    # Agent SSE streaming hook
│   ├── use-event-source.ts  # Event bus SSE listener hook
│   └── client/              # Auto-generated API client (hey-api/openapi-ts)
│       ├── types.gen.ts     # Generated TypeScript types from OpenAPI
│       ├── sdk.gen.ts       # Generated SDK functions
│       ├── client.gen.ts    # Client configuration
│       └── @tanstack/       # Generated React Query hooks
└── types/
    └── tensorspace.d.ts     # TensorSpace TypeScript declarations
```

### 6.2 State Management

- **React Query** (`@tanstack/react-query`): All server state (sessions, steps, diagnostics, health) is fetched and cached via auto-generated React Query hooks. Stale time is 60 seconds.
- **SSE invalidation** (`useEventSource`): When backend events arrive, specific React Query caches are invalidated, triggering automatic refetches.
- **Local state** (`useState`): UI-only state (selected project, selected session, panel visibility, agent messages).

### 6.3 Component Map

#### `ProjectsClient.tsx` — Main Orchestrator

The root component that orchestrates the entire dashboard:
- **Project selection**: Left sidebar with project list, create project functionality
- **Session management**: Lists training sessions for selected project, maps API types to panel types
- **Event source**: Connects to SSE stream filtered by selected project
- **Layout**: Multi-panel layout with session list, training details, charts, agent, and sustainability panels
- **Layer health scoring**: Computes per-layer health scores from diagnostic issues, using severity weights and metric-specific penalty functions

#### `ProjectTrainingPanels.tsx` — Session Details

Large component (~1489 lines) containing multiple sub-panels:
- `TrainSessionPanel`: Session metadata (device, PyTorch version, config, status, stop/resume controls)
- `TrainStepList`: Scrollable list of epoch steps with loss/throughput summaries
- `ModelPanel`: Model architecture display (module tree, hyperparameters, layer graph)
- `SessionIssuesPanel`: Diagnostic issues with severity badges, filtering, and "fix prompt" generation
- `BottomTerminalPanel`: Agent terminal + log viewer in a split panel

#### `StepsDashboard.tsx` — Training Charts

Comprehensive charting via Recharts:
- Loss curves (train mean/min/max/std, validation loss, validation accuracy)
- Throughput (samples/sec, tokens/sec, batches/sec)
- Memory usage (process RSS, CUDA allocated, CUDA peak)
- System metrics (CPU %, RAM %)
- Epoch duration bar chart
- Profiler top ops table
- Summary statistics table

#### `SustainabilityPanel.tsx` — Green-AI Metrics

Displays sustainability insights from the latest diagnostic run:
- Parameter efficiency grade (A–F)
- Total CO₂ and energy consumption
- Wasted compute (epochs, CO₂, cost in EUR)
- Optimal stopping point
- Problematic layers (dead, vanishing gradients, frozen, redundant)
- CO₂ equivalents (e.g., "equivalent to X km driven")

#### `ThreeScene.tsx` — 3D Visualization

Embeds a TensorSpace.js model inside the React dashboard:
- Parses model architecture from the API
- Maps layers (Conv2d, MaxPool2d, Flatten, Dense, Output) to TensorSpace primitives
- Color-tints layers by sustainability score
- Hover tooltips with layer info and issues
- Supports both CNN and generic architectures

#### `AgentTerminalPanel.tsx` — Agent Chat

A terminal-style chat interface for the Atlas agent:
- Quick action buttons (health, sustainability, architecture, diagnose)
- Markdown rendering of agent responses via `react-markdown`
- Belief state display bar (severity, sustainability grade, confidence, revision count)
- Streaming response display

#### `ProactiveInsightBanner.tsx` — Auto-Alerts

Banner notifications for proactive agent insights:
- Color-coded by severity (critical=red, warning=amber, watch=blue, healthy=green)
- "Updated Assessment" badge when the agent revises prior analysis
- "Ask Atlas to explain" button for follow-up

### 6.4 Real-Time Updates

**`useEventSource` hook** (`lib/use-event-source.ts`):
- Connects to `GET /events/stream` with project/session ID filters
- Listens for all event types defined in `INVALIDATION_MAP`
- On event: invalidates matching React Query caches by `_id` prefix
- Auto-reconnects with exponential backoff (1s → 30s max)
- 15-second keepalive from server prevents proxy timeouts

**`useAgentChat` hook** (`lib/use-agent-chat.ts`):
- Manages agent conversation state (messages, belief state, loading, error)
- Sends user messages to `POST /agent/chat`
- Parses SSE stream (message, belief, error, done events)
- Supports abort and clear history
- Passes accumulated belief state back on each turn

### 6.5 3D Visualization

The frontend includes TensorSpace-based 3D rendering:
- Parses `Model.architecture` from the API into layer definitions
- Creates TensorSpace Sequential model with proper layer mapping
- Grayscale 28×28 input (optimized for MNIST-like datasets)
- Supports Conv2d, MaxPool2d, Dense, Output1d layers
- Auto-adds Flatten between spatial and dense layers
- Color-codes layers by sustainability score (green=healthy → red=problematic)
- Hover tooltips show layer ID, type, and parameters

---

## 7. Neural Network Observer

### 7.1 Observer Class

**File:** `neural_network/observer.py` (~1869 lines)

The Observer is a comprehensive PyTorch training monitor that:
1. Creates a session in the backend on initialization
2. Registers model architecture and hyperparameters
3. Collects per-batch and per-epoch metrics
4. Syncs telemetry to the backend via HTTP
5. Captures console and error logs from Python's logging system
6. Tracks carbon emissions via CodeCarbon
7. Exports a complete JSON report for offline analysis

#### Lifecycle

```
Observer.__init__()      → Creates session in backend
    │
    ├── register_model()  → Installs hooks, sends architecture
    │
    ├── (Training loop)
    │   ├── should_profile()  → Check if this step should be profiled
    │   ├── profile_step()    → Run forward+backward with torch.profiler
    │   ├── step()            → Record batch metrics (loss, throughput, memory)
    │   └── flush()           → Aggregate epoch, send to backend, reset
    │
    ├── export()             → Write JSON report to file
    └── close()              → Finalize session, remove hooks, stop carbon tracker
```

### 7.2 ObserverConfig

Every telemetry channel can be independently enabled/disabled:

| Config Key | Default | Purpose |
|-----------|---------|---------|
| `track_profiler` | `True` | PyTorch profiler (op-level timing) |
| `track_memory` | `True` | RAM + CUDA memory snapshots |
| `track_throughput` | `True` | Samples/sec, tokens/sec |
| `track_loss` | `True` | Mean, std, min, max per epoch |
| `track_console_logs` | `True` | Capture Python logging |
| `track_error_logs` | `True` | Capture stderr |
| `track_hyperparameters` | `True` | Record HP dict |
| `track_layer_graph` | `True` | Detailed layer-graph with dimensions |
| `track_system_resources` | `True` | CPU %, RAM % |
| `track_sustainability` | `True` | Layer efficiency, marginal loss |
| `track_layer_health` | `True` | Activation/gradient hooks |
| `track_carbon_emissions` | `True` | CO₂ via CodeCarbon |
| `profile_at_step` | `0` | Which step to profile |
| `profile_every_n_steps` | `None` | Profile periodically |
| `pending_timeout` | `3600` | Auto-stop if pending too long |
| `carbon_country_iso` | `"IRL"` | Country code for grid carbon intensity |

### 7.3 Telemetry Channels

Each epoch's `TrainStep` payload includes:

```json
{
  "step_index": 3,
  "timestamp": "2026-02-22T10:15:00",
  "duration_seconds": 12.5,
  "loss": {
    "train_mean": 2.345,
    "train_std": 0.12,
    "train_min": 2.10,
    "train_max": 2.58,
    "count": 64,
    "val": { "val_loss": 2.56, "val_acc": 0.31 }
  },
  "throughput": {
    "samples_per_sec": 512.3,
    "tokens_per_sec": 16384.0,
    "batches_per_sec": 8.0,
    "samples_processed": 6400
  },
  "memory": {
    "process_rss_mb": 1024.5,
    "cuda_allocated_mb": 512.0,
    "cuda_peak_allocated_mb": 768.0
  },
  "system": {
    "cpu_percent": 45.2,
    "ram_percent": 62.1
  },
  "profiler": {
    "top_ops": [...],
    "per_layer": [...],
    "total_cpu_time_us": 12500000,
    "total_fwd_us": 4000000,
    "total_bwd_us": 8000000
  },
  "layer_health": {
    "per_layer": {
      "blocks.0.sa.heads.0.key": {
        "activation_mean": 0.001,
        "activation_std": 0.15,
        "activation_var_of_means": 0.02,
        "gradient_norm_mean": 0.005,
        "weight_sparsity": 0.12,
        "is_dead": false,
        "has_vanishing_gradients": false
      }
    },
    "activation_correlations": [...]
  },
  "sustainability": {
    "layer_efficiency": [...],
    "marginal_loss_improvement": 0.03,
    "cumulative_loss_improvement": 0.45
  },
  "carbon_emissions": {
    "epoch_co2_kg": 0.000025,
    "epoch_energy_kwh": 0.00015,
    "power_draw_watts": 42.5,
    "cumulative_co2_kg": 0.0001,
    "cumulative_energy_kwh": 0.0006
  },
  "log_counts": {
    "console": 12,
    "error": 0
  }
}
```

### 7.4 Report Schema (Pydantic)

**File:** `neural_network/schema.py` (~456 lines)

Root model: `ObserverReport`

```
ObserverReport
├── session: Session
│   ├── project_id, run_id, run_name
│   ├── started_at, ended_at
│   ├── device, cuda_available, pytorch_version
│   └── config: SessionConfig
├── hyperparameters: Dict[str, Any]
├── model_architecture: ModelArchitecture
│   ├── layers: Dict[str, LayerInfo]           # flat map of leaf modules
│   ├── module_tree: ModuleTree                 # recursive nn.Module tree
│   └── layer_graph: LayerGraph                 # detailed graph (when enabled)
│       ├── nodes: List[GraphNode]              # per-node: type, dims, params, buffers
│       ├── edges: List[GraphEdge]              # containment + data_flow edges
│       ├── sequential_path: List[str]          # topological order of leaves
│       └── dimension_flow: List[DimensionFlow] # tensor shape at each stage
├── steps: List[StepRecord]                     # per-epoch telemetry
└── summary: TrainingSummary
```

**GraphNode** contains rich per-layer metadata:
- Linear: `in_features`, `out_features`, `has_bias`, `weight_shape`
- Embedding: `num_embeddings`, `embedding_dim`
- Conv2d: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`
- LayerNorm: `normalized_shape`, `eps`
- Dropout: `p`
- Attention: `embed_dim`, `num_heads`, `head_dim`
- Per-parameter: shape, numel, dtype, mean, std, norm

### 7.5 Backend Sync Protocol

The Observer communicates with the backend via HTTP requests:

| Lifecycle Event | HTTP Call | When |
|----------------|-----------|------|
| Session creation | `POST /sessions/project/{project_id}` | `__init__()` |
| Model registration | `POST /sessions/{id}/model` | `register_model()` |
| Step telemetry | `POST /sessions/{id}/step` | `flush()` |
| Log entries | `POST /sessions/{id}/logs` | `flush()` (batched) |
| Status polling | `GET /sessions/{id}/status` | `flush()` (after step) |
| Session update | `PATCH /sessions/{id}` | `close()` |

**Pause/stop interaction**: After each `flush()`, the Observer polls the session status. If the backend has set it to `pending` (due to critical diagnostics), the Observer waits with exponential backoff (up to `pending_timeout` seconds). It can also handle `stop`/`resume` actions from the dashboard.

---

## 8. Standalone Visualization (Vite + TensorSpace)

**Directory:** `visualization/`

A separate Vite application for 3D neural network architecture visualization:

- **`src/api.js`**: Parses Observer JSON reports into `{ name, layers: [{id, type, params}] }`. Contains `paramsFor()` which derives renderer-specific parameters from hyperparameters and the flat layer dict.
- **`src/main.js`**: Creates a TensorSpace Sequential model, maps parsed layers to TensorSpace layer types (Conv2d, Pooling2d, Dense, Output1d), and renders the 3D scene.
- **`mock-report.json`**: GPT architecture (128-emb, 4-layer, 4-head, child speech data)
- **`cnn-mock-report.json`**: CNN architecture (SmallCNN for MNIST)

**Layer mapping**: The visualizer handles the `FeedFoward` typo from the PyTorch model (both `FeedForward` and `FeedFoward` spellings are supported).

---

## 9. Cross-Cutting Concerns

### Error Handling

- Backend: FastAPI `HTTPException` with appropriate status codes (404, 409, 500, 502)
- Agent: Try Crusoe → catch → fallback to Anthropic → catch → 502
- Observer: Never crashes the training loop; all errors are caught and logged
- Frontend: React Query error states, SSE reconnection with backoff

### Security

- API keys stored in `.env` (not committed)
- CORS restricted to specific origins
- `CRUSOE_API_KEY` and `ANTHROPIC_API_KEY` for LLM providers
- No authentication on API endpoints (hackathon scope)

### Performance Considerations

- SQLite with WAL mode (via SQLModel/SQLAlchemy)
- Log batching in Observer (configurable interval + max size)
- SSE queue backpressure (drop oldest event if queue full)
- Tool call result truncation (12KB max per tool result for LLM context)
- Sustainability data capped at 50 layer efficiency snapshots per report

### Known Technical Debt

- `FeedFoward` typo from PyTorch model propagates through entire stack
- Fix prompt cache is in-memory only (lost on restart)
- No authentication or rate limiting
- SQLite is single-writer (not suitable for high concurrency)
- Background diagnostics can race with new step registrations

---

## 10. Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **SQLite** instead of PostgreSQL | Hackathon simplicity; single file, zero configuration, portable |
| **SSE** instead of WebSocket | Simpler protocol for unidirectional server→client push; sufficient for cache invalidation |
| **In-memory event bus** instead of Redis | Single-process deployment; no external dependencies needed |
| **Tool-calling agent** instead of RAG | Direct database access is more accurate than embedding-based retrieval for numerical training data |
| **Crusoe + Anthropic dual-provider** | Hackathon API availability; Crusoe is primary (free tier), Anthropic is fallback |
| **Observer as HTTP client** (not library import) | Decouples training environment from backend; Observer can run in any Jupyter kernel while backend runs separately |
| **Heuristic diagnostics** instead of ML-based anomaly detection | Interpretable, debuggable, no training data required; thresholds are tunable constants |
| **Belief state in agent** | Implements the incident.io adaptive agent challenge requirement; enables self-revision tracking |
| **CodeCarbon** for emissions | Established library for ML carbon tracking; supports both online (grid data) and offline modes |
| **TensorSpace** for 3D visualization | Purpose-built for neural network visualization; supports Conv2d, Dense, and other layer types natively |
| **`dim()` helper inlined** per visualization file | Intentional; avoids cross-module coupling in standalone ES module factory functions |
| **Auto-generated API client** (hey-api/openapi-ts) | Type-safe frontend-backend contract; regenerated from OpenAPI spec |
