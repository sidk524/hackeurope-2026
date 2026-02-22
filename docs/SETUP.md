# Setup Guide — Atlas ML Training Observatory

> **Project:** Atlas — ML Training Observatory & Green-AI Advisor  
> **Event:** HackEurope 2026  
> **Last updated:** 2026-02-22

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Repository Structure](#2-repository-structure)
3. [Environment Variables](#3-environment-variables)
4. [Backend Setup](#4-backend-setup)
5. [Frontend Setup](#5-frontend-setup)
6. [Neural Network Observer Setup](#6-neural-network-observer-setup)
7. [Standalone Visualization Setup](#7-standalone-visualization-setup)
8. [Running the Full Stack](#8-running-the-full-stack)
9. [MCP Server Setup (Claude Desktop)](#9-mcp-server-setup-claude-desktop)
10. [Database Management](#10-database-management)
11. [API Client Regeneration](#11-api-client-regeneration)
12. [Troubleshooting](#12-troubleshooting)
13. [Development Workflow](#13-development-workflow)

---

## 1. Prerequisites

### Required Software

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| **Python** | 3.10+ (3.12 recommended) | Backend server, Observer, neural network training |
| **Node.js** | 18+ | Frontend (Next.js) and standalone visualization (Vite) |
| **npm** or **yarn** | npm 9+ / yarn 1.22+ | JavaScript package management |
| **pip** | Latest | Python package management |
| **Git** | 2.30+ | Version control |

### Optional Software

| Tool | Purpose |
|------|---------|
| **pyenv** | Python version management (used in this workspace) |
| **CUDA** | GPU-accelerated PyTorch training (Observer detects automatically) |
| **Docker** | Not currently configured, but could be used for deployment |

### Required API Keys

| Key | Provider | Required For |
|-----|----------|-------------|
| `CRUSOE_API_KEY` | [Crusoe Cloud](https://hackeurope.crusoecloud.com) | LLM proxy (`/llm`) and primary agent provider |
| `ANTHROPIC_API_KEY` | [Anthropic](https://anthropic.com) | Agent fallback provider (optional but recommended) |

---

## 2. Repository Structure

```
hackeurope-2026/
├── CLAUDE.md                    # AI assistant guidance
├── README.md                    # Project README
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md          # Deep technical architecture reference
│   ├── SETUP.md                 # This file
│   ├── DIAGNOSTICS_GUIDE.md     # Complete diagnostics engine guide
│   └── OBSERVER_AND_BACKEND_GUIDE.md  # Observer & backend integration guide
├── changes/                     # Change proposals / implementation plans
│   └── greenai-sustainability-diagnostics.md
├── backend/                     # FastAPI backend server
├── frontend/                    # Next.js 16 dashboard
├── neural_network/              # PyTorch Observer + training notebooks
├── visualization/               # Standalone Vite + TensorSpace 3D visualizer
└── tensorspace/                 # (unused placeholder)
```

---

## 3. Environment Variables

### Backend (`backend/.env`)

Create `backend/.env` from the example:

```bash
cd backend
cp .env.example .env
```

Edit `.env` with your keys:

```dotenv
# Required: Crusoe Cloud API key for Qwen3-235B LLM access
CRUSOE_API_KEY=your-crusoe-api-key-here

# Optional: Anthropic API key for Claude fallback
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional: Custom database URL (defaults to SQLite ./app.db)
# DATABASE_URL=sqlite:///./app.db

# Optional: Comma-separated CORS origins (defaults to localhost:3000,localhost:5173)
# CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Frontend

The frontend uses a single environment variable (defaults to `http://localhost:8000`):

```dotenv
# Optional: Override backend URL
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

This can be set in `frontend/.env.local` if needed.

---

## 4. Backend Setup

### Step 1: Navigate to the backend directory

```bash
cd backend
```

### Step 2: (Optional) Set Python version with pyenv

```bash
pyenv shell 3.10.4
# or
pyenv shell 3.12.0
```

### Step 3: Create and activate a virtual environment

```bash
# Create venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.\.venv\Scripts\activate.bat

# Activate (Linux/macOS)
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `fastapi[standard]==0.129.1` — Web framework with uvicorn
- `sqlmodel==0.0.35` — ORM (Pydantic + SQLAlchemy)
- `fastmcp>=2.0.0` — Model Context Protocol server
- `python-dotenv==1.0.1` — .env file loading
- `openai>=1.0.0` — OpenAI-compatible API client (for Crusoe)
- `anthropic>=0.40.0` — Anthropic API client (for Claude fallback)

### Step 5: Initialize the database

The first time you run the server, the SQLite database is created automatically. However, if the schema has changed, run Alembic migrations:

```bash
# Apply all migrations
alembic upgrade head
```

If starting fresh (no existing `app.db`):

```bash
# The database and tables will be auto-created on first server start
# Alternatively, create via Alembic:
alembic upgrade head
```

### Step 6: Set up environment variables

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Step 7: Start the server

```bash
uvicorn main:fastapi --reload
```

The server starts at `http://localhost:8000`.

**Verify it's running:**
```bash
# Check the auto-generated API docs
curl http://localhost:8000/docs        # Swagger UI
curl http://localhost:8000/redoc       # ReDoc
curl http://localhost:8000/openapi.json # OpenAPI spec
```

---

## 5. Frontend Setup

### Step 1: Navigate to the frontend directory

```bash
cd frontend
```

### Step 2: Install dependencies

```bash
npm install
# or
yarn install
```

**Key dependencies:**
- `next@16.1.6` — React framework
- `react@19.2.3` — UI library
- `@tanstack/react-query@5.74.7` — Server state management
- `recharts@2.15.4` — Charting library
- `react-markdown@10.1.0` — Markdown rendering (for agent responses)
- `tensorspace@0.6.1` — 3D neural network visualization
- `three@0.86.0` — 3D rendering engine (required by TensorSpace)
- `lucide-react@0.575.0` — Icon library
- `@hey-api/client-fetch@0.10.0` — Auto-generated API client
- `radix-ui@1.4.3` — Accessible UI primitives

### Step 3: Start the development server

```bash
npm run dev
```

The app starts at `http://localhost:3000`.

### Step 4: Build for production (verification)

```bash
npm run build
```

This is the correctness check — there is no linter or test runner configured. If the build succeeds, the code is valid.

---

## 6. Neural Network Observer Setup

### Step 1: Navigate to the neural_network directory

```bash
cd neural_network
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `torch[cuda]` — PyTorch (with optional CUDA support)
- `psutil` — System resource monitoring
- `pydantic` — Data validation (for report schema)
- `torchvision` — Computer vision datasets and transforms
- `codecarbon` — Carbon emissions tracking

> **Note:** If you don't have a CUDA GPU, PyTorch will install without CUDA support. The Observer detects this automatically and falls back to CPU.

### Step 3: Ensure the backend is running

The Observer communicates with the backend via HTTP. Make sure `uvicorn main:fastapi --reload` is running on port 8000.

### Step 4: Open a training notebook

The sample notebooks are in `neural_network/sample/`:

| Notebook | Model | Dataset | Purpose |
|----------|-------|---------|---------|
| `mini_gpt.ipynb` | Mini GPT Transformer | Child speech data | Transformer architecture demo |
| `small_cnn.ipynb` | SmallCNN | MNIST | CNN architecture demo |
| `buggy_cnn.ipynb` | SmallCNN (with bugs) | MNIST | Demonstrates diagnostic issue detection |

**Start Jupyter:**

```bash
jupyter notebook
# or
jupyter lab
```

### Step 5: Run the training notebook

Each notebook follows this pattern:

```python
from observer import Observer, ObserverConfig

# Configure Observer
config = ObserverConfig(
    track_profiler=True,
    track_layer_health=True,
    track_sustainability=True,
    track_carbon_emissions=True,
    profile_at_step=0,
)

# Create Observer (creates session in backend)
obs = Observer(project_id=1, config=config, run_name="my-experiment")

# Register model (installs hooks, sends architecture)
obs.register_model(model)
obs.log_hyperparameters({
    "batch_size": 64,
    "learning_rate": 3e-4,
    "n_embd": 128,
    # ...
})

# Training loop
for epoch in range(num_epochs):
    for step, (x, y) in enumerate(dataloader):
        if obs.should_profile(step):
            logits, loss = obs.profile_step(model, x, y)
        else:
            logits, loss = model(x, y)
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        obs.step(step, loss, batch_size=x.size(0), seq_length=x.size(1))
    
    # End of epoch - flush to backend
    obs.flush(val_metrics={"val_loss": val_loss, "val_acc": val_acc})

# Finish
obs.export("observer_reports/my-run.json")  # Optional: save to file
obs.close()
```

### Step 6: View results in the dashboard

1. Open `http://localhost:3000` in your browser
2. Select or create a project
3. Watch the training session appear in real-time
4. View loss curves, system metrics, and diagnostics as they update

---

## 7. Standalone Visualization Setup

### Step 1: Navigate to the visualization directory

```bash
cd visualization
```

### Step 2: Install dependencies

```bash
npm install
```

### Step 3: Start the development server

```bash
npm run dev
```

The visualizer starts at `http://localhost:5173`.

**Other commands:**
```bash
npm run build    # Production build (correctness check)
npm run preview  # Preview production build
```

### Data Source

By default, the visualizer loads `cnn-mock-report.json` (a real Observer export from a SmallCNN MNIST run). To change the data source, edit `src/api.js`:

```javascript
// Currently:
import mockReport from '../cnn-mock-report.json';

// To use GPT report:
import mockReport from '../mock-report.json';
```

---

## 8. Running the Full Stack

Here is the complete sequence to run all components simultaneously:

### Terminal 1: Backend

```bash
cd backend
# Activate your Python virtualenv
uvicorn main:fastapi --reload
# → http://localhost:8000
```

### Terminal 2: Frontend

```bash
cd frontend
npm run dev
# → http://localhost:3000
```

### Terminal 3: Training (optional)

```bash
cd neural_network
jupyter notebook
# Open and run a training notebook
```

### Terminal 4: Standalone Visualization (optional)

```bash
cd visualization
npm run dev
# → http://localhost:5173
```

### Quick Verification

1. **Backend**: Visit `http://localhost:8000/docs` — should show Swagger UI
2. **Frontend**: Visit `http://localhost:3000` — should show dashboard (may be empty if no projects exist)
3. **Create a project**: Use the "+" button in the dashboard sidebar, or:
   ```bash
   curl -X POST http://localhost:8000/projects/ \
     -H "Content-Type: application/json" \
     -d '{"name": "My First Project"}'
   ```
4. **Start training**: Run a notebook cell-by-cell. Watch the dashboard update in real-time.

---

## 9. MCP Server Setup (Claude Desktop)

The backend includes an MCP (Model Context Protocol) server for integration with Claude Desktop.

### Running via stdio (for Claude Desktop)

```bash
cd backend
python mcp_server.py
```

### Running via SSE

```bash
cd backend
python mcp_server.py --transport sse --port 8100
```

### Claude Desktop Configuration

Add to your Claude Desktop config (`~/.config/claude/claude_desktop_config.json` or equivalent):

```json
{
  "mcpServers": {
    "ml-diagnostics": {
      "command": "python",
      "args": ["path/to/backend/mcp_server.py"],
      "env": {
        "DATABASE_URL": "sqlite:///path/to/backend/app.db"
      }
    }
  }
}
```

---

## 10. Database Management

### Database Location

The default SQLite database is at `backend/app.db`.

### Alembic Migrations

```bash
cd backend

# Show current migration status
alembic current

# Apply all pending migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# Create a new migration (after modifying models.py)
alembic revision --autogenerate -m "describe your change"
```

### Migration History

| Revision | Description |
|----------|-------------|
| `488e1970ad6c` | Initial schema — Project, TrainSession, Model, TrainStep, SessionLog |
| `a1b2c3d4e5f6` | Add DiagnosticRun and DiagnosticIssue tables |
| `e57e1f3af050` | Add sustainability columns (layer_health, sustainability, carbon_emissions, log_counts) to TrainStep |

### Reset Database

To start completely fresh:

```bash
cd backend
rm app.db
alembic upgrade head
```

Or simply delete `app.db` and restart the server — tables will be recreated.

---

## 11. API Client Regeneration

The frontend uses auto-generated TypeScript types and React Query hooks from the backend's OpenAPI spec.

### Prerequisites

The backend must be running on port 8000.

### Regenerate

```bash
cd frontend

# 1. Fetch the latest OpenAPI spec and regenerate TypeScript client
npm run build:openapi

# This runs:
#   curl http://localhost:8000/openapi.json -o openapi.json
#   yarn openapi-ts
```

The generated files are in `frontend/src/lib/client/`:
- `types.gen.ts` — TypeScript types for all API models
- `sdk.gen.ts` — SDK functions for all API endpoints
- `client.gen.ts` — Client configuration
- `@tanstack/react-query.gen.ts` — React Query hooks

---

## 12. Troubleshooting

### Backend won't start

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'fastapi'` | Dependencies not installed | `pip install -r requirements.txt` |
| `CRUSOE_API_KEY is not set` | Missing .env file | Copy `.env.example` to `.env` and add your key |
| `alembic.util.exc.CommandError` | Database schema mismatch | `alembic upgrade head` or delete `app.db` |
| `Address already in use` | Port 8000 is occupied | Kill the existing process or use `--port 8001` |

### Frontend won't start

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `Module not found` | Dependencies not installed | `npm install` |
| `ECONNREFUSED localhost:8000` | Backend not running | Start backend first |
| Build errors with types | Stale API client | Run `npm run build:openapi` with backend running |

### Observer can't connect

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ConnectionRefusedError` | Backend not running | Start backend on port 8000 |
| `404 Not Found` on session creation | No project exists | Create a project first via API or dashboard |
| `Session is not running` | Session paused by diagnostics | Use dashboard to resume, or increase `pending_timeout` |

### Agent not responding

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Both LLM providers failed" | Missing API keys | Set `CRUSOE_API_KEY` and/or `ANTHROPIC_API_KEY` |
| Agent gives empty responses | Qwen3 think blocks not stripped | This should be handled automatically; check logs |
| Tool calls timing out | SQLite contention | Restart backend (background tasks may have stalled) |

### Visualization issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Blank TensorSpace canvas | WebGL not supported | Try a different browser (Chrome recommended) |
| "TensorSpace render failed" | Invalid model data | Check that `cnn-mock-report.json` is valid JSON |
| No tooltip on hover | TensorSpace hasn't initialized | Wait for model init callback |

---

## 13. Development Workflow

### Making Frontend Changes

1. Edit components in `frontend/src/`
2. Next.js hot-reloads automatically
3. Run `npm run build` to verify no TypeScript errors before committing

### Making Backend Changes

1. Edit Python files in `backend/`
2. Uvicorn with `--reload` auto-restarts on file changes
3. If you modify `models.py`:
   - Create a migration: `alembic revision --autogenerate -m "description"`
   - Apply it: `alembic upgrade head`
4. If you add/change API endpoints:
   - Regenerate the frontend client: `cd frontend && npm run build:openapi`

### Making Observer Changes

1. Edit `neural_network/observer.py` or `schema.py`
2. Restart your Jupyter kernel to pick up changes
3. Re-run notebook cells

### Adding New Diagnostic Checks

1. Add your check function to `backend/diagnostics/engine.py`
2. Register it in `run_diagnostics()` (in the appropriate check list for the architecture type)
3. If adding a new `IssueCategory`, update the enum in `backend/models.py` and create a migration
4. Test by triggering diagnostics: `POST /diagnostics/sessions/{id}/run`

### Adding New Visualization Layer Types

For the standalone visualizer (`visualization/`):
1. Create `visualization/src/models/yourlayer.js`
2. Import and register in `visualization/src/main.js`

For the frontend TensorSpace view:
1. Update the layer mapping in `frontend/src/app/components/ThreeScene.tsx`

### Coding Conventions

| Language | Style |
|----------|-------|
| Python | Standard library conventions; type hints; dataclasses for data containers |
| TypeScript | Strict mode; PascalCase components; camelCase hooks/utils; 2-space indent |
| CSS | Tailwind v4 utility classes; `clsx` + `tailwind-merge` for composition |
| Git | Short, descriptive commit messages; no Conventional Commits requirement |
