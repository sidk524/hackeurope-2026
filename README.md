<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python" alt="Python">
<img src="https://img.shields.io/badge/Node.js-18+-green?style=flat-square&logo=node.js" alt="Node.js">
<img src="https://img.shields.io/badge/Next.js-16-black?style=flat-square&logo=next.js" alt="Next.js">
<img src="https://img.shields.io/badge/FastAPI-0.129-green?style=flat-square&logo=fastapi" alt="FastAPI">
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
<img src="https://img.shields.io/badge/HackEurope-2026-purple?style=flat-square" alt="HackEurope 2026">

<h1>Atlas — ML Training Observatory</h1>

<p><strong>Real-time diagnostics, adaptive AI agent, and sustainability scoring for machine learning training runs</strong></p>

</div>

---

## What is Atlas?

Atlas is an end-to-end ML training observatory that **watches your model train in real-time** and tells you what's going wrong, how to fix it, and how much carbon you're burning. It combines:

- A **Python Observer** that hooks into any PyTorch training loop to collect telemetry (loss, gradients, activations, memory, profiler data, carbon emissions)
- A **FastAPI backend** that stores training data, runs 30+ diagnostic heuristics, and powers an adaptive AI agent
- A **Next.js dashboard** with live-updating charts, 3D neural network visualization, and a conversational AI terminal
- A **standalone 3D visualizer** powered by TensorSpace for exploring network architecture

---

## Table of Contents

- [What is Atlas?](#what-is-atlas)
- [Architecture Overview](#architecture-overview)
- [Components](#components)
  - [Neural Network Observer](#1-neural-network-observer-neural_network)
  - [Backend Server](#2-backend-server-backend)
  - [Frontend Dashboard](#3-frontend-dashboard-frontend)
  - [Standalone Visualization](#4-standalone-visualization-visualization)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)
- [Documentation](#documentation)
- [Acknowledgments](#acknowledgments)

---

## Architecture Overview

```
┌─────────────────────┐       HTTP/POST        ┌──────────────────────────┐
│  PyTorch Training   │ ──────────────────────► │   FastAPI Backend        │
│  + Observer hooks   │    steps, logs, arch    │   :8000                  │
│  + CodeCarbon       │ ◄────── sessions ────── │                          │
│  neural_network/    │                         │  ┌─────────────────────┐ │
└─────────────────────┘                         │  │ SQLite (app.db)     │ │
                                                │  │ Projects, Sessions  │ │
                                                │  │ Steps, Diagnostics  │ │
                                                │  └─────────────────────┘ │
                                                │                          │
                                                │  ┌─────────────────────┐ │
                                                │  │ Diagnostics Engine  │ │
                                                │  │ 30+ heuristic rules │ │
                                                │  │ Health score 0–100  │ │
                                                │  └─────────────────────┘ │
                                                │                          │
                                                │  ┌─────────────────────┐ │
                                                │  │ Agent (Atlas)       │ │
                                                │  │ Crusoe + Anthropic  │ │
                                                │  │ 10 tools, SSE chat  │ │
                                                │  └─────────────────────┘ │
                                                │                          │
                                                │  ┌─────────────────────┐ │
                                                │  │ Event Bus (SSE)     │ │
                                                │  │ Real-time push      │ │
                                                │  └─────────────────────┘ │
                                                │                          │
                                                │  ┌─────────────────────┐ │
                                                │  │ MCP Server          │ │
                                                │  │ Claude Desktop      │ │
                                                │  └─────────────────────┘ │
                                                └──────────┬───────────────┘
                                                           │ SSE + REST
                                                           ▼
                                                ┌──────────────────────────┐
                                                │   Next.js Dashboard      │
                                                │   :3000                  │
                                                │                          │
                                                │  • Live loss/metric      │
                                                │    charts (Recharts)     │
                                                │  • 3D network viz        │
                                                │    (TensorSpace)         │
                                                │  • Agent chat terminal   │
                                                │  • Sustainability panel  │
                                                │  • Diagnostics view      │
                                                │  • Proactive insights    │
                                                └──────────────────────────┘

                                                ┌──────────────────────────┐
                                                │   Standalone Visualizer  │
                                                │   :5173  (Vite)          │
                                                │   TensorSpace + Three.js │
                                                └──────────────────────────┘
```

---

## Components

### 1. Neural Network Observer (`neural_network/`)

**What:** A drop-in Python class that wraps any PyTorch training loop to collect per-step and per-epoch telemetry.

**Why:** Training ML models is a black box. The Observer turns it into a glass box — you see every gradient distribution, every memory spike, every watt consumed, without modifying your model code.

**How it works:**
- `Observer.register_model(model)` installs forward/backward hooks on every `nn.Module` leaf to capture activations, gradients, and weight statistics
- Each training step, `obs.step()` records loss, batch size, timing, memory usage, and optionally PyTorch profiler data
- `obs.flush()` at epoch boundaries ships everything to the backend via HTTP POST
- CodeCarbon integration tracks real-time energy consumption and CO₂ emissions
- On completion, `obs.export()` dumps a full Pydantic-validated JSON report

**Key files:**

| File | Purpose |
|------|---------|
| `observer.py` | Main Observer class (~1,870 lines) with ObserverConfig for toggling channels |
| `schema.py` | Pydantic models defining the ObserverReport JSON schema (~456 lines) |
| `sample/mini_gpt.ipynb` | Transformer (Mini GPT) training on child speech data |
| `sample/small_cnn.ipynb` | CNN training on MNIST |
| `sample/buggy_cnn.ipynb` | Intentionally buggy CNN that triggers diagnostic issues |

**Configurable channels:**

```python
ObserverConfig(
    track_profiler=True,         # PyTorch profiler (expensive)
    track_activations=True,      # Per-layer activation stats (expensive)
    track_attention_entropy=True, # Attention head entropy (expensive)
    track_layer_health=True,     # Gradient/weight health per layer
    track_sustainability=True,   # Energy/carbon tracking
    track_carbon_emissions=True, # CodeCarbon integration
)
```

---

### 2. Backend Server (`backend/`)

**What:** A FastAPI application that serves as the central hub — storing training data, running diagnostics, hosting the AI agent, and pushing real-time events to the frontend.

**Why:** The Observer collects raw data, but data without analysis is just numbers. The backend transforms telemetry into actionable intelligence through its diagnostics engine and AI agent.

**How it works:**

#### Six API Router Groups

| Router | Endpoint Prefix | Purpose |
|--------|----------------|---------|
| **Projects** | `/projects/` | CRUD for project containers; aggregates session statuses |
| **Sessions** | `/sessions/` | Session lifecycle (create → running → done); step ingestion; log management |
| **Diagnostics** | `/diagnostics/` | Run 30+ heuristic checks; retrieve issues, health scores, and trends |
| **Events** | `/events/` | SSE streaming — pushes `step_logged`, `session_started`, `diagnostics_complete`, etc. |
| **LLM** | `/llm/` | Transparent proxy to Crusoe Cloud (Qwen3-235B-A22B) |
| **Agent** | `/agent/` | Conversational AI with tool-calling loop; proactive analysis; belief-state tracking |

#### Diagnostics Engine (30+ checks)

The engine in `diagnostics/engine.py` (~1,807 lines) runs heuristic checks across categories:

- **Loss anomalies** — Divergence, NaN/Inf, plateau detection, oscillation
- **Gradient health** — Vanishing/exploding gradients, gradient norm spikes
- **System** — GPU memory pressure, CPU bottleneck, batch size inefficiency
- **Profiler** — Slow data loading, excessive CPU time, kernel inefficiency
- **Sustainability** — Carbon intensity, energy waste, idle GPU detection, efficiency grading (A–F)
- **Architecture** — Dead ReLU, attention entropy collapse, embedding scale, dropout misconfiguration

Each check produces an `Issue` with severity (critical/warning/info), a human-readable message, and a machine-readable `fix_prompt`. Final health score: `max(0, 100 - Σ severity_weights)`.

#### Adaptive Agent (Atlas)

The agent implements a dual-provider tool-calling loop:
1. **Primary:** Crusoe Cloud → Qwen3-235B-A22B-Instruct (via OpenAI SDK)
2. **Fallback:** Anthropic → Claude Sonnet 4

It has access to 10 tools (get_session_detail, run_diagnostics, get_sustainability_report, compare_sessions, etc.) and uses a **belief-state protocol** — after each response, it outputs a JSON block tracking its current hypothesis, confidence level, and open questions. This enables self-revision across multi-turn conversations.

#### Event Bus

An in-memory pub/sub system that pushes 8 event types via Server-Sent Events (SSE):
`step_logged`, `epoch_completed`, `session_started`, `session_completed`, `session_status_changed`, `diagnostics_complete`, `log_added`, `model_architecture_updated`

#### MCP Server

A FastMCP server exposing the same diagnostic tools for Claude Desktop integration, available via stdio or SSE transport.

---

### 3. Frontend Dashboard (`frontend/`)

**What:** A Next.js 16 single-page application that provides a rich visual interface for monitoring training runs.

**Why:** ML practitioners need immediate visual feedback during training. The dashboard turns raw step data into interactive charts, a conversational agent, and 3D architecture visualization — all updating in real-time via SSE.

**How it works:**

#### Main Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| `ProjectsClient.tsx` | ~920 | Main orchestrator — project sidebar, session list, panel switching |
| `ProjectTrainingPanels.tsx` | ~1,489 | Tabbed detail view — Training, Architecture, Diagnostics, Sustainability, Agent |
| `StepsDashboard.tsx` | ~1,408 | Recharts dashboard — loss curves, gradient norms, learning rate, memory usage, throughput |
| `AgentTerminalPanel.tsx` | ~403 | Chat terminal for the Atlas agent with SSE streaming |
| `SustainabilityPanel.tsx` | ~307 | Green-AI panel — carbon emissions, energy usage, sustainability grade (A–F), recommendations |
| `ThreeScene.tsx` | ~598 | TensorSpace 3D neural network visualization |
| `ProactiveInsightBanner.tsx` | ~160 | Auto-triggered agent analysis banner with streaming results |
| `ProjectTrendChart.tsx` | ~333 | Cross-session comparison charts |

#### Real-Time Updates

- `use-event-source.ts` subscribes to `/events/stream?project_id=X` via SSE
- Each event type maps to React Query cache keys for automatic invalidation
- Charts and panels re-render immediately when new data arrives — no polling

#### State Management

- **Server state:** React Query (`@tanstack/react-query`) for all API data
- **Client state:** React hooks (`useState`, `useReducer`) for UI state
- **No Redux/Zustand** — React Query cache is the single source of truth

---

### 4. Standalone Visualization (`visualization/`)

**What:** A standalone Vite + TensorSpace application that renders interactive 3D neural network architectures from Observer JSON reports.

**Why:** Understanding model architecture visually helps debug shape mismatches, identify bottlenecks, and communicate architecture decisions. This component works independently of the main dashboard for quick architecture exploration.

**How it works:**

- `api.js` parses Observer JSON reports and extracts layer types, sizes, and connections
- `main.js` builds a TensorSpace `Sequential` model with appropriate layer types (Dense, Conv2d, Pooling2d)
- Supports both GPT (Transformer) and CNN architecture reports
- Hover tooltips show layer details (name, parameters, percentage of total)
- Renders at `http://localhost:5173`

---

## Key Features

### Real-Time Training Monitoring
- Live loss curves, gradient norms, learning rates, and throughput charts
- Per-epoch and per-step granularity
- Automatic axis scaling and multi-metric overlays

### 30+ Automatic Diagnostics
- Detects divergence, plateaus, vanishing gradients, dead ReLUs, attention collapse
- Health score 0–100 with severity-weighted deductions
- Machine-readable fix prompts the agent can act on

### Adaptive AI Agent
- Conversational interface powered by Qwen3-235B (Crusoe Cloud)
- 10 diagnostic tools for deep analysis
- Belief-state reasoning: tracks hypothesis confidence and open questions
- Proactive analysis: auto-triggers when session completes, offers unsolicited insights

### Green-AI Sustainability
- Real-time energy tracking via CodeCarbon
- CO₂ emissions with regional grid carbon intensity
- Sustainability grades (A–F) with actionable recommendations
- Wasted compute detection (idle GPU, unnecessary epochs, oversized batches)
- EU ETS cost estimation at €50/ton CO₂

### 3D Architecture Visualization
- TensorSpace-powered interactive 3D model viewer
- Supports Transformer (GPT) and CNN architectures
- Integrated in the dashboard + available as standalone tool

### MCP Integration
- Full diagnostic toolkit available in Claude Desktop
- stdio and SSE transport modes

---

## Quick Start

### Prerequisites

- Python 3.10+ and Node.js 18+
- A Crusoe Cloud API key ([hackeurope.crusoecloud.com](https://hackeurope.crusoecloud.com))

### 1. Backend

```bash
cd backend
cp .env.example .env        # Add your CRUSOE_API_KEY
pip install -r requirements.txt
alembic upgrade head
uvicorn main:fastapi --reload   # → http://localhost:8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev                     # → http://localhost:3000
```

### 3. Train a Model

```bash
cd neural_network
pip install -r requirements.txt
jupyter notebook                # Open sample/small_cnn.ipynb and run all cells
```

Watch the dashboard update in real-time as your model trains.

> For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

---

## Tech Stack

### Backend

| Technology | Version | Role |
|------------|---------|------|
| FastAPI | 0.129.1 | Web framework |
| SQLModel | 0.0.35 | ORM (Pydantic + SQLAlchemy) |
| SQLite | — | Database |
| Alembic | — | Schema migrations |
| FastMCP | 2.0+ | Model Context Protocol server |
| OpenAI SDK | 1.0+ | Crusoe Cloud LLM client |
| Anthropic SDK | 0.40+ | Claude fallback |

### Frontend

| Technology | Version | Role |
|------------|---------|------|
| Next.js | 16.1.6 | React framework |
| React | 19.2.3 | UI library |
| TypeScript | — | Type safety |
| Tailwind CSS | v4 | Styling |
| React Query | 5.74.7 | Server state management |
| Recharts | 2.15.4 | Charting |
| TensorSpace | 0.6.1 | 3D neural network visualization |
| Three.js | 0.86.0 | 3D engine (TensorSpace dependency) |
| shadcn/ui | — | Accessible UI primitives |

### Neural Network

| Technology | Role |
|------------|------|
| PyTorch | Deep learning framework |
| CodeCarbon | Carbon emissions tracking |
| Pydantic | Report schema validation |
| psutil | System resource monitoring |
| torchvision | Datasets and transforms |

### Standalone Visualization

| Technology | Version | Role |
|------------|---------|------|
| Vite | 7.3.1 | Build tool |
| TensorSpace | 0.6.1 | 3D neural network rendering |
| Three.js | 0.183.1 | 3D engine |
| Express | 5.2.1 | Static file server |

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Deep technical architecture reference — every component, every data flow |
| [docs/SETUP.md](docs/SETUP.md) | Step-by-step installation and configuration guide |
| [docs/DIAGNOSTICS_GUIDE.md](docs/DIAGNOSTICS_GUIDE.md) | Complete diagnostics engine reference — all 30+ checks |
| [docs/OBSERVER_AND_BACKEND_GUIDE.md](docs/OBSERVER_AND_BACKEND_GUIDE.md) | Observer API and backend integration guide |
| [CLAUDE.md](CLAUDE.md) | AI assistant guidance / project conventions |
| [changes/greenai-sustainability-diagnostics.md](changes/greenai-sustainability-diagnostics.md) | Green-AI implementation plan |

---

## Acknowledgments

Built with:

- [FastAPI](https://fastapi.tiangolo.com/) — Backend web framework
- [Next.js](https://nextjs.org/) — Frontend framework
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [TensorSpace](https://tensorspace.org/) — 3D neural network visualization
- [Three.js](https://threejs.org/) — 3D rendering engine
- [Recharts](https://recharts.org/) — React charting library
- [CodeCarbon](https://codecarbon.io/) — Carbon emissions tracker
- [Crusoe Cloud](https://crusoecloud.com/) — Sustainable GPU cloud + Qwen3-235B LLM
- [Anthropic](https://www.anthropic.com/) — Claude API (agent fallback)
- [FastMCP](https://github.com/jlowin/fastmcp) — Model Context Protocol server
- [shadcn/ui](https://ui.shadcn.com/) — Accessible UI components

---

<div align="center">
<strong>HackEurope 2026</strong>
</div>
