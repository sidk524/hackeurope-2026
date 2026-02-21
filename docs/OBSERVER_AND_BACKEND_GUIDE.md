# Observer & Backend Integration Guide

> **For Junior Developers** - This guide explains everything about the Observer system and how it communicates with the Backend server.

## Table of Contents

1. [What is the Observer?](#1-what-is-the-observer)
2. [The Big Picture - How Everything Connects](#2-the-big-picture---how-everything-connects)
3. [ObserverConfig - All the Settings](#3-observerconfig---all-the-settings)
4. [Creating an Observer](#4-creating-an-observer)
5. [The Training Loop Workflow](#5-the-training-loop-workflow)
6. [Backend API Endpoints](#6-backend-api-endpoints)
7. [Data Structures](#7-data-structures)
8. [Session Lifecycle & Status](#8-session-lifecycle--status)
9. [Diagnostics System](#9-diagnostics-system)
10. [Complete Code Examples](#10-complete-code-examples)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What is the Observer?

The **Observer** is a Python class that "watches" your PyTorch neural network while it's training. Think of it like a fitness tracker for your AI model - it measures everything that's happening:

| What it Tracks | Why It's Useful |
|----------------|-----------------|
| **Loss** | Is your model learning? Is the loss going down? |
| **Throughput** | How fast is training? (samples/second, tokens/second) |
| **Memory** | How much RAM/GPU memory is being used? |
| **Profiler** | Which operations are slow? Where are the bottlenecks? |
| **Layer Health** | Are any layers "dead"? Are gradients vanishing? |
| **Carbon Emissions** | How much CO2 is your training producing? |
| **Logs** | What messages are being printed during training? |

### Simple Analogy

Imagine you're baking a cake:
- The **model** is your recipe
- **Training** is the actual baking process
- The **Observer** is like having someone watch the oven, check the temperature, time each step, and write down notes

---

## 2. The Big Picture - How Everything Connects

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR JUPYTER NOTEBOOK                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │        PyTorch Training Loop                                     │   │
│  │                                                                  │   │
│  │   for step, (x, y) in enumerate(dataloader):                    │   │
│  │       logits, loss = model(x, y)                                │   │
│  │       loss.backward()                                           │   │
│  │       optimizer.step()                                          │   │
│  │       observer.step(step, loss)  ◄───────────────┐              │   │
│  │       observer.flush()           ◄───────────────┤ OBSERVER     │   │
│  └──────────────────────────────────────────────────┼──────────────┘   │
└─────────────────────────────────────────────────────┼───────────────────┘
                                                      │
                                                      │ HTTP Requests
                                                      │ (POST, GET)
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         BACKEND SERVER (FastAPI)                         │
│                         http://localhost:8000                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                                      │   │
│  │    POST /sessions/project/{id}     → Create training session     │   │
│  │    POST /sessions/{id}/model       → Register model architecture │   │
│  │    POST /sessions/{id}/step        → Log training step data      │   │
│  │    POST /sessions/{id}/log         → Push console/error logs     │   │
│  │    GET  /sessions/{id}/status      → Check session status        │   │
│  │    POST /sessions/{id}/action      → Stop/resume session         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                      │                                   │
│                                      ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SQLite Database (app.db)                      │   │
│  │                                                                  │   │
│  │  Tables:                                                         │   │
│  │    - Project       (groups of training sessions)                 │   │
│  │    - TrainSession  (one training run)                            │   │
│  │    - Model         (architecture + hyperparameters)              │   │
│  │    - TrainStep     (metrics for each epoch/step)                 │   │
│  │    - SessionLog    (console and error logs)                      │   │
│  │    - DiagnosticRun (analysis results)                            │   │
│  │    - DiagnosticIssue (problems found by analysis)                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

1. **You start training** → Observer creates a session in the backend
2. **You register a model** → Observer sends architecture to backend
3. **Each training step** → Observer collects metrics
4. **You call `flush()`** → Observer sends all metrics to backend
5. **Backend runs diagnostics** → Checks for problems in your training
6. **Backend updates status** → `running`, `pending`, `stopped`, etc.

---

## 3. ObserverConfig - All the Settings

The `ObserverConfig` class controls what the Observer tracks. Here's every setting explained:

```python
from dataclasses import dataclass
from typing import Optional
import logging

@dataclass
class ObserverConfig:
    """Controls which telemetry channels the Observer records."""
    
    # ═══════════════════════════════════════════════════════════════
    # CORE TRACKING - The basic stuff
    # ═══════════════════════════════════════════════════════════════
    
    track_profiler: bool = True
    # Should we measure how long each operation takes?
    # Example: "matrix multiplication took 15ms, softmax took 3ms"
    
    track_memory: bool = True
    # Should we track RAM and GPU memory usage?
    # Example: "GPU using 2.5GB of 8GB total"
    
    track_throughput: bool = True
    # Should we measure training speed?
    # Example: "Processing 1500 tokens per second"
    
    track_loss: bool = True
    # Should we track the loss value?
    # Example: "Loss went from 4.5 to 2.1"
    
    # ═══════════════════════════════════════════════════════════════
    # LOGGING - Capturing print statements and errors
    # ═══════════════════════════════════════════════════════════════
    
    track_console_logs: bool = True
    # Capture all INFO-level and above log messages
    # These go to the backend in real-time
    
    track_error_logs: bool = True
    # Capture all WARNING, ERROR, CRITICAL log messages
    # Useful for debugging problems later
    
    track_hyperparameters: bool = True
    # Store the hyperparameters (learning rate, batch size, etc.)
    
    # ═══════════════════════════════════════════════════════════════
    # ARCHITECTURE - Understanding your model's structure
    # ═══════════════════════════════════════════════════════════════
    
    track_layer_graph: bool = True
    # Build a detailed graph of all layers in your model
    # Used by the 3D visualization frontend
    # Includes: layer types, dimensions, connections
    
    # ═══════════════════════════════════════════════════════════════
    # SYSTEM - Computer resource monitoring
    # ═══════════════════════════════════════════════════════════════
    
    track_system_resources: bool = True
    # Track CPU usage, RAM usage, GPU info
    # Example: "CPU at 45%, RAM at 67%"
    
    # ═══════════════════════════════════════════════════════════════
    # PROFILER CONFIGURATION - Fine-tuning performance measurement
    # ═══════════════════════════════════════════════════════════════
    
    profile_at_step: Optional[int] = 0
    # Which step should we profile? (0 = first step)
    # Profiling is expensive, so we only do it once by default
    
    profile_every_n_steps: Optional[int] = None
    # Profile every N steps instead of just once
    # Example: profile_every_n_steps=10 → profile at step 0, 10, 20, 30...
    
    profiler_record_shapes: bool = True
    # Record the shape of input/output tensors
    # Example: "Linear layer input: [32, 128], output: [32, 256]"
    
    profiler_profile_memory: bool = True
    # Track memory allocation during profiling
    
    profiler_with_stack: bool = False
    # Include call stack info (which function called what)
    # SLOW - only enable if debugging performance issues
    
    profiler_top_n_ops: int = 20
    # How many slowest operations to report
    
    profiler_group_by_stack_n: int = 0
    # Group operations by call stack depth (0 = disabled)
    
    profiler_top_n_stacks: int = 20
    # How many call stacks to show
    
    # ═══════════════════════════════════════════════════════════════
    # GREEN AI / SUSTAINABILITY - Health and efficiency tracking
    # ═══════════════════════════════════════════════════════════════
    
    track_sustainability: bool = True
    # Track efficiency metrics:
    # - Layer efficiency (compute time vs parameters)
    # - Marginal loss improvement
    # - Compute costs
    
    track_layer_health: bool = True
    # Monitor each layer for problems:
    # - Dead neurons (always output 0)
    # - Vanishing gradients (gradients too small)
    # - Exploding activations
    # WARNING: Adds hooks to every layer, some overhead
    
    layer_health_zero_threshold: float = 1e-6
    # Values below this are considered "near-zero"
    # Used to detect dead weights/activations
    
    # ═══════════════════════════════════════════════════════════════
    # CARBON EMISSIONS - Environmental impact tracking
    # ═══════════════════════════════════════════════════════════════
    
    track_carbon_emissions: bool = True
    # Track CO2 emissions and energy consumption
    # Requires: pip install codecarbon
    
    carbon_tracker_mode: str = "online"
    # "online"  → Gets real-time electricity grid data
    # "offline" → Uses static country-based estimates
    
    carbon_country_iso: str = "IRL"
    # ISO 3166-1 alpha-3 country code
    # Used for carbon intensity calculations
    # Examples: "USA", "GBR", "DEU", "FRA", "IRL"
    
    # ═══════════════════════════════════════════════════════════════
    # BACKEND COMMUNICATION
    # ═══════════════════════════════════════════════════════════════
    
    pending_timeout: float = 39.0
    # How long to wait (seconds) when session is "pending"
    # The backend might pause training if issues are detected
    # Observer will auto-stop after this timeout
    
    # ═══════════════════════════════════════════════════════════════
    # DEBUGGING
    # ═══════════════════════════════════════════════════════════════
    
    log_level: int = logging.INFO
    # Observer's own logging level
    # logging.DEBUG   = Lots of messages
    # logging.INFO    = Normal (default)
    # logging.WARNING = Only problems
```

### Example Configurations

```python
# MINIMAL CONFIG - Just loss and throughput, no heavy stuff
minimal_config = ObserverConfig(
    track_profiler=False,           # No profiling
    track_layer_health=False,       # No layer monitoring
    track_carbon_emissions=False,   # No carbon tracking
    track_layer_graph=False,        # No architecture graph
)

# DEBUGGING CONFIG - Maximum information
debug_config = ObserverConfig(
    track_profiler=True,
    profile_every_n_steps=5,        # Profile often
    profiler_with_stack=True,       # Include call stacks
    track_layer_health=True,
    track_sustainability=True,
    log_level=logging.DEBUG,        # Verbose logging
)

# PRODUCTION CONFIG - Balanced for real training
prod_config = ObserverConfig(
    track_profiler=True,
    profile_at_step=0,              # Profile only first step
    track_layer_health=True,
    track_sustainability=True,
    track_carbon_emissions=True,
)
```

---

## 4. Creating an Observer

### Basic Creation

```python
from observer import Observer, ObserverConfig

# Step 1: Create a configuration (or use defaults)
config = ObserverConfig(
    track_profiler=True,
    track_layer_health=True,
)

# Step 2: Create the Observer
# - project_id: Which project this training belongs to (integer)
# - config: The configuration object
# - run_name: A name for this training run (optional)
observer = Observer(
    project_id=1,
    config=config,
    run_name="my_first_training",
)
```

### What Happens When You Create an Observer?

```
Observer() called
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 1. Generate run_id = "{project_id}_{run_name}"                 │
│    Example: "1_my_first_training"                              │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. Create session metadata dictionary:                          │
│    {                                                            │
│      "project_id": 1,                                           │
│      "run_id": "1_my_first_training",                           │
│      "started_at": "2026-02-21T10:30:00",                       │
│      "device": "NVIDIA GeForce RTX 4090",                       │
│      "cuda_available": true,                                    │
│      "pytorch_version": "2.1.0",                                │
│      ...                                                        │
│    }                                                            │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. Initialize empty storage:                                    │
│    - hyperparameters = {}                                       │
│    - model_architecture = {}                                    │
│    - step_data = []                                             │
│    - console_logs = []                                          │
│    - error_logs = []                                            │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. Attach log capture handlers                                  │
│    (intercepts Python's logging system)                         │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ 5. HTTP POST to backend:                                        │
│    URL: http://localhost:8000/sessions/project/1                │
│    Body: {session metadata}                                     │
│    Response: { "id": 42 }  ← Backend session ID                 │
└────────────────────────────────────────────────────────────────┘
```

### The Backend Response

When the Observer creates a session, the backend:

1. **Creates a new `TrainSession` record** in the database
2. **Returns the session ID** (e.g., `42`)
3. The Observer stores this as `_backend_session_id`

All future communications use this ID.

---

## 5. The Training Loop Workflow

Here's the complete workflow with code:

### Step-by-Step Breakdown

```python
import torch
from observer import Observer, ObserverConfig

# ══════════════════════════════════════════════════════════════════
# PHASE 1: SETUP
# ══════════════════════════════════════════════════════════════════

# 1. Create your model (regular PyTorch)
model = MyNeuralNetwork()

# 2. Create the observer
config = ObserverConfig(track_profiler=True)
obs = Observer(project_id=1, config=config, run_name="experiment_1")

# 3. Log hyperparameters (the settings for your training)
obs.log_hyperparameters({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "optimizer": "Adam",
    "model_type": "Transformer",
})
# This just stores them locally in obs.hyperparameters

# 4. Register the model (THIS SENDS DATA TO BACKEND!)
obs.register_model(model)
# What happens:
#   - Counts all parameters (weights)
#   - Maps every layer (name, type, size)
#   - Builds the layer graph for visualization
#   - Attaches hooks for layer health monitoring
#   - POST to /sessions/{id}/model with architecture data

# ══════════════════════════════════════════════════════════════════
# PHASE 2: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

for epoch in range(num_epochs):
    for step, (inputs, targets) in enumerate(dataloader):
        
        # Should we profile this step?
        if obs.should_profile(step):
            # Use the profiler (wraps forward + backward)
            logits, loss = obs.profile_step(model, inputs, targets)
        else:
            # Regular training step
            logits, loss = model(inputs, targets)
            loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # 5. Tell observer about this step
        obs.step(
            step=step,
            loss=loss,
            batch_size=inputs.size(0),
            seq_length=inputs.size(1),  # for language models
        )
        # This DOES NOT send to backend yet!
        # It just accumulates data in memory:
        #   - Adds loss to _step_batch_losses
        #   - Adds timing to _step_batch_times
        #   - Increments _step_samples_processed
        #   - Increments _step_tokens_processed
    
    # ══════════════════════════════════════════════════════════════
    # END OF EPOCH - FLUSH!
    # ══════════════════════════════════════════════════════════════
    
    # 6. Flush the accumulated data (THIS SENDS TO BACKEND!)
    step_record = obs.flush(val_metrics={"val_loss": val_loss})
    # What happens:
    #   - Calculates duration since last flush
    #   - Computes loss statistics (mean, min, max, std)
    #   - Computes throughput (samples/sec, tokens/sec)
    #   - Snapshots memory usage
    #   - Snapshots system resources
    #   - Computes layer health metrics
    #   - Computes sustainability metrics
    #   - Gets carbon emissions (if tracking)
    #   - POST to /sessions/{id}/step with all this data
    #   - GET /sessions/{id}/status to check if we should continue
    
    print(f"Epoch {epoch} done! Loss: {step_record['loss']['train_mean']}")

# ══════════════════════════════════════════════════════════════════
# PHASE 3: CLEANUP
# ══════════════════════════════════════════════════════════════════

# 7. Export to JSON file (local backup)
obs.export("observer_reports/my_run.json")

# 8. Close the observer (cleanup)
obs.close()
# What happens:
#   - Removes log capture handlers
#   - Removes layer health hooks
#   - Stops carbon tracker
#   - Sets ended_at timestamp
```

### Visual Timeline

```
Time →
═══════════════════════════════════════════════════════════════════════════

START
  │
  ├─► Observer() created
  │     └─► POST /sessions/project/1 (create session in backend)
  │
  ├─► log_hyperparameters({...}) 
  │     └─► Stored locally only
  │
  ├─► register_model(model)
  │     └─► POST /sessions/42/model (send architecture to backend)
  │
  │   ┌─────────────────── EPOCH 0 ───────────────────┐
  │   │                                                │
  │   │  step(0, loss)  ──► Accumulate in memory      │
  │   │  step(1, loss)  ──► Accumulate in memory      │
  │   │  step(2, loss)  ──► Accumulate in memory      │
  │   │  ...                                           │
  │   │  step(N, loss)  ──► Accumulate in memory      │
  │   │                                                │
  │   │  flush() ─────────────────────────────────────►│ POST /sessions/42/step
  │   │           ◄───────────────────────────────────│ GET /sessions/42/status
  │   │                                                │
  │   └────────────────────────────────────────────────┘
  │
  │   ┌─────────────────── EPOCH 1 ───────────────────┐
  │   │  (same pattern repeats)                        │
  │   └────────────────────────────────────────────────┘
  │
  ├─► export("path/to/file.json")
  │     └─► Writes JSON to disk
  │
  └─► close()
        └─► Cleanup
```

---

## 6. Backend API Endpoints

The Observer communicates with these endpoints:

### POST `/sessions/project/{project_id}`
**Purpose:** Create a new training session

```python
# What Observer sends:
{
    "run_id": "1_my_experiment",
    "run_name": "my_experiment",
    "started_at": "2026-02-21T10:30:00.000000",
    "device": "NVIDIA GeForce RTX 4090",
    "cuda_available": true,
    "pytorch_version": "2.1.0",
    "config": {
        "track_profiler": true,
        "track_memory": true,
        ...
    },
    "status": "running"
}

# What backend returns:
{
    "id": 42,
    "project_id": 1,
    "run_id": "1_my_experiment",
    "run_name": "my_experiment",
    "started_at": "2026-02-21T10:30:00.000000",
    "status": "running",
    ...
}
```

### POST `/sessions/{session_id}/model`
**Purpose:** Register model architecture

```python
# What Observer sends:
{
    "architecture": {
        "total_parameters": 1500000,
        "trainable_parameters": 1500000,
        "frozen_parameters": 0,
        "num_parameter_layers": 12,
        "layers": {
            "embedding.tok_emb": {
                "type": "Embedding",
                "parameters": 65536,
                "pct_of_total": 4.37
            },
            "blocks.0.attn.c_attn": {
                "type": "Linear",
                "parameters": 49152,
                "pct_of_total": 3.28
            },
            ...
        },
        "module_tree": {
            "type": "GPT",
            "children": {
                "embedding": {
                    "type": "Embedding",
                    ...
                },
                "blocks": {
                    "type": "ModuleList",
                    "children": {...}
                }
            }
        },
        "layer_graph": {
            "nodes": [...],
            "edges": [...],
            "sequential_path": [...],
            "dimension_flow": [...]
        }
    },
    "hyperparameters": {
        "learning_rate": 0.001,
        "batch_size": 32,
        ...
    }
}
```

### POST `/sessions/{session_id}/step`
**Purpose:** Record metrics for a training step/epoch

```python
# What Observer sends:
{
    "step_index": 0,
    "timestamp": "2026-02-21T10:31:45.123456",
    "duration_seconds": 12.5,
    "loss": {
        "train_mean": 4.234567,
        "train_min": 3.901234,
        "train_max": 4.567890,
        "train_std": 0.123456,
        "num_batches": 100,
        "val": {"val_loss": 4.1}
    },
    "throughput": {
        "samples_processed": 3200,
        "tokens_processed": 409600,
        "samples_per_second": 256.0,
        "tokens_per_second": 32768.0
    },
    "profiler": {
        "total_cpu_time_ms": 8500.234,
        "total_cuda_time_ms": 3200.567,
        "top_operations": [
            {
                "name": "aten::addmm",
                "calls": 1200,
                "cpu_time_ms": 2500.0
            },
            ...
        ],
        "per_layer": [
            {
                "layer": "blocks.0.ffn.fc1",
                "fwd_us": 120000,
                "bwd_us": 180000,
                "total_us": 300000,
                "pct_of_total": 15.5
            },
            ...
        ]
    },
    "memory": {
        "cuda_allocated_mb": 2048.5,
        "cuda_peak_allocated_mb": 3072.0,
        "process_rss_mb": 4096.0
    },
    "system": {
        "cpu_percent": 45.2,
        "ram_percent": 67.8,
        "gpu_name": "NVIDIA GeForce RTX 4090"
    },
    "layer_health": {
        "layers": {
            "blocks.0.ffn.fc1": {
                "activation_mean": 0.5,
                "activation_std": 0.2,
                "gradient_norm_mean": 0.001,
                "weight_sparsity": 0.05,
                "is_dead": false,
                "has_vanishing_gradients": false
            },
            ...
        }
    },
    "sustainability": {
        "layer_efficiency": [...],
        "marginal_loss": {...},
        "step_compute_cost": {...},
        "cumulative_compute": {...}
    }
}
```

### GET `/sessions/{session_id}/status`
**Purpose:** Check if training should continue

```python
# What Observer receives:
"running"     # Continue training
"pending"     # Pause! Diagnostics found issues, waiting for user
"stopped"     # Stop training
"completed"   # Training finished normally
"failed"      # Something went wrong
"analyzing"   # Backend is running diagnostics
```

### POST `/sessions/{session_id}/log`
**Purpose:** Push real-time log messages

```python
# What Observer sends:
{
    "ts": "2026-02-21T10:30:15.123456",
    "level": "INFO",
    "msg": "[observer] step=0 loss=4.567890 batch_size=32",
    "module": "observer",
    "lineno": 123,
    "kind": "console"
}
```

### POST `/sessions/{session_id}/action`
**Purpose:** Resume or stop a pending session

```python
# User decides to resume:
{"action": "resume"}  # → sets status to "running"

# User decides to stop:
{"action": "stop"}    # → sets status to "stopped"
```

---

## 7. Data Structures

### Database Models (Backend)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PROJECT                                     │
│  id: 1                                                                   │
│  name: "My Language Model Project"                                       │
│  created_at: 2026-02-20T09:00:00                                        │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         TRAINSESSION                               │  │
│  │  id: 42                                                            │  │
│  │  project_id: 1                                                     │  │
│  │  run_id: "1_my_experiment"                                         │  │
│  │  status: "running" | "pending" | "stopped" | "completed"           │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                          MODEL                               │  │  │
│  │  │  id: 100                                                     │  │  │
│  │  │  session_id: 42                                              │  │  │
│  │  │  architecture: {...}                                         │  │  │
│  │  │  hyperparameters: {...}                                      │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                       TRAINSTEP (×N)                         │  │  │
│  │  │  id: 200                                                     │  │  │
│  │  │  session_id: 42                                              │  │  │
│  │  │  step_index: 0                                               │  │  │
│  │  │  loss: {...}                                                 │  │  │
│  │  │  throughput: {...}                                           │  │  │
│  │  │  memory: {...}                                               │  │  │
│  │  │  ...                                                         │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                      SESSIONLOG (×M)                         │  │  │
│  │  │  id: 300                                                     │  │  │
│  │  │  session_id: 42                                              │  │  │
│  │  │  level: "INFO"                                               │  │  │
│  │  │  msg: "Step 0 completed"                                     │  │  │
│  │  │  kind: "console" | "error"                                   │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                     DIAGNOSTICRUN                            │  │  │
│  │  │  id: 400                                                     │  │  │
│  │  │  session_id: 42                                              │  │  │
│  │  │  health_score: 85                                            │  │  │
│  │  │  issue_count: 3                                              │  │  │
│  │  │                                                              │  │  │
│  │  │  ┌───────────────────────────────────────────────────────┐  │  │  │
│  │  │  │                 DIAGNOSTICISSUE (×K)                   │  │  │  │
│  │  │  │  severity: "critical" | "warning" | "info"            │  │  │  │
│  │  │  │  category: "loss" | "memory" | "sustainability" | ... │  │  │  │
│  │  │  │  title: "Loss not decreasing"                         │  │  │  │
│  │  │  │  description: "Training loss has plateaued..."        │  │  │  │
│  │  │  │  suggestion: "Try lowering learning rate..."          │  │  │  │
│  │  │  └───────────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Session Status Enum

```python
class SessionStatus(str, Enum):
    running = "running"      # Normal training in progress
    completed = "completed"  # Training finished successfully
    failed = "failed"        # Training crashed/errored
    pending = "pending"      # PAUSED - waiting for user input
    analyzing = "analyzing"  # Backend running diagnostics
    stopped = "stopped"      # User manually stopped
```

---

## 8. Session Lifecycle & Status

### The Status State Machine

```
                                    ┌──────────────┐
                                    │   CREATED    │
                                    └──────┬───────┘
                                           │
                            Observer() creates session
                                           │
                                           ▼
                    ┌──────────────────────────────────────────┐
                    │                RUNNING                    │
                    │                                          │
                    │  Training is happening normally.          │
                    │  Observer sends steps, backend analyzes.  │
                    └───────────────────┬──────────────────────┘
                                        │
            ┌───────────────────────────┼───────────────────────────┐
            │                           │                           │
            ▼                           ▼                           ▼
   ┌─────────────────┐        ┌─────────────────┐         ┌─────────────────┐
   │    ANALYZING    │        │     PENDING     │         │    COMPLETED    │
   │                 │        │                 │         │                 │
   │ Backend running │        │ Issues found!   │         │ Training done   │
   │ diagnostics on  │───────►│ Waiting for     │         │ successfully    │
   │ the latest step │        │ user decision   │         │                 │
   └─────────────────┘        └────────┬────────┘         └─────────────────┘
            │                          │
            │                          │
            │              ┌───────────┴───────────┐
            │              │                       │
            │              ▼                       ▼
            │     ┌─────────────────┐     ┌─────────────────┐
            │     │     STOPPED     │     │     RUNNING     │
            │     │                 │     │                 │
            │     │ User chose to   │     │ User chose to   │
            │     │ stop training   │     │ resume training │
            │     │                 │     │                 │
            │     └─────────────────┘     └─────────────────┘
            │
            └─────────────────────────────────────────────────►  RUNNING
                      (diagnostics passed, no critical issues)
```

### When Does the Session Go to "PENDING"?

The backend sets status to `pending` when it finds:

1. **Any CRITICAL issue** (e.g., loss is NaN, memory overflow)
2. **3+ WARNINGS on the same layer** (e.g., vanishing gradients, dead neurons)

### Observer's Response to PENDING

After each `flush()`, the Observer polls `/sessions/{id}/status`:

```python
def _await_backend_status(self, rec: Dict[str, Any]) -> None:
    """Await the status of the session with exponential backoff timeout."""
    elapsed = 0.0
    delay = 1.0  # Start with 1 second

    while True:
        status = self._poll_backend_status(rec)
        
        if status == "running":
            return  # Continue training!
        
        if status in ("completed", "stopped"):
            raise RuntimeError("Training was stopped")
        
        if status == "failed":
            raise RuntimeError("Training marked as failed")
        
        if status in ("paused", "pending"):
            # Wait with exponential backoff
            time.sleep(delay)
            elapsed += delay
            delay *= 2  # Double the wait time
            
            if elapsed > self.config.pending_timeout:
                # Timeout! Save checkpoint and stop
                self._save_checkpoint()
                self._stop_backend_session()
                raise RuntimeError("Timeout waiting for pending status")
```

---

## 9. Diagnostics System

### What the Diagnostics Engine Checks

When a step is registered, the backend runs diagnostics in a background task:

```python
# From backend/routers/sessions.py
@router.post("/{session_id}/step")
def register_step(session_id, step_data, background_tasks):
    # 1. Save the step
    step_db = TrainStep(...)
    
    # 2. Set status to "analyzing"
    train_session.status = SessionStatus.analyzing
    
    # 3. Run diagnostics in background
    background_tasks.add_task(_run_step_diagnostics, session_id)
    
    return step_db
```

### Categories of Issues

| Category | What It Checks | Example Issue |
|----------|---------------|---------------|
| `loss` | Loss values | "Loss is NaN", "Loss not decreasing" |
| `throughput` | Training speed | "Very slow training speed" |
| `memory` | RAM/GPU usage | "GPU memory near capacity" |
| `profiler` | Performance | "Forward/backward ratio unusual" |
| `logs` | Log messages | "Too many error logs" |
| `system` | System resources | "CPU throttling detected" |
| `architecture` | Model structure | "Very deep network, may have vanishing gradients" |
| `sustainability` | Green AI | "Wasted compute after convergence" |

### Issue Severity Levels

```python
class IssueSeverity(str, Enum):
    critical = "critical"  # MUST address - training may be broken
    warning = "warning"    # SHOULD address - training may be suboptimal  
    info = "info"          # FYI - interesting but not problematic
```

### Example Diagnostic Issues

```python
# CRITICAL: Training is broken
DiagnosticIssue(
    severity="critical",
    category="loss",
    title="Loss is NaN",
    description="The loss value became NaN at epoch 5, which means "
                "your gradients exploded or there's a numerical issue.",
    suggestion="Try: 1) Lower learning rate, 2) Add gradient clipping, "
               "3) Check for division by zero in your model",
    epoch_index=5,
    metric_value={"loss": float("nan")}
)

# WARNING: Something might be wrong
DiagnosticIssue(
    severity="warning",
    category="sustainability",
    title="Vanishing gradients in layer",
    description="Layer 'blocks.3.ffn.fc2' has gradient norm < 1e-7, "
                "which means it's barely learning anything.",
    suggestion="Consider: 1) Use residual connections, 2) Try different "
               "initialization, 3) Check layer is connected properly",
    layer_id="blocks.3.ffn.fc2",
    metric_value={"gradient_norm": 1e-9}
)

# INFO: Just letting you know
DiagnosticIssue(
    severity="info",
    category="throughput",
    title="Training speed changed",
    description="Training throughput increased by 20% in epoch 3. "
                "This is normal if data complexity varies.",
    epoch_index=3,
    metric_value={"tokens_per_second": 45000}
)
```

---

## 10. Complete Code Examples

### Example 1: Basic CNN Image Classifier

```python
import torch
import torch.nn as nn
from observer import Observer, ObserverConfig

# ══════════════════════════════════════════════════════════════════
# 1. Define your model (regular PyTorch)
# ══════════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 16 * 16, num_classes)
    
    def forward(self, x, targets=None):
        x = self.relu1(self.conv1(x))
        x = self.pool(x)
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits, targets)
        
        return logits, loss

# ══════════════════════════════════════════════════════════════════
# 2. Set up observer
# ══════════════════════════════════════════════════════════════════

config = ObserverConfig(
    track_profiler=True,
    profile_at_step=0,       # Profile first step only
    track_layer_health=True,
    track_sustainability=True,
    track_carbon_emissions=False,  # Disable if codecarbon not installed
)

observer = Observer(
    project_id=1,            # Must match a project in the backend database
    config=config,
    run_name="cnn_cifar10",
)

# ══════════════════════════════════════════════════════════════════
# 3. Log hyperparameters
# ══════════════════════════════════════════════════════════════════

observer.log_hyperparameters({
    "model": "SimpleCNN",
    "dataset": "CIFAR-10",
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "optimizer": "Adam",
})

# ══════════════════════════════════════════════════════════════════
# 4. Create model and register it
# ══════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

observer.register_model(model)  # Sends architecture to backend

# ══════════════════════════════════════════════════════════════════
# 5. Training loop
# ══════════════════════════════════════════════════════════════════

# Fake data for this example
train_loader = [
    (torch.randn(64, 3, 32, 32).to(device), torch.randint(0, 10, (64,)).to(device))
    for _ in range(100)
]
val_loader = [
    (torch.randn(64, 3, 32, 32).to(device), torch.randint(0, 10, (64,)).to(device))
    for _ in range(10)
]

for epoch in range(10):
    model.train()
    
    for step, (images, labels) in enumerate(train_loader):
        # Profiling decision
        if observer.should_profile(step):
            logits, loss = observer.profile_step(model, images, labels)
        else:
            logits, loss = model(images, labels)
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Record the step (accumulates in memory)
        observer.step(
            step=step,
            loss=loss,
            batch_size=images.size(0),
        )
    
    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for images, labels in val_loader:
            _, val_loss = model(images, labels)
            val_losses.append(val_loss.item())
    
    avg_val_loss = sum(val_losses) / len(val_losses)
    
    # FLUSH! This sends data to backend
    step_record = observer.flush(val_metrics={"val_loss": avg_val_loss})
    
    print(f"Epoch {epoch}: "
          f"train_loss={step_record['loss']['train_mean']:.4f}, "
          f"val_loss={avg_val_loss:.4f}")

# ══════════════════════════════════════════════════════════════════
# 6. Cleanup
# ══════════════════════════════════════════════════════════════════

observer.export("observer_reports/cnn_cifar10.json")
observer.close()
```

### Example 2: Language Model (GPT-style)

```python
import torch
import torch.nn as nn
from observer import Observer, ObserverConfig

# ══════════════════════════════════════════════════════════════════
# Model definition (simplified GPT)
# ══════════════════════════════════════════════════════════════════

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        x = tok_emb + pos_emb
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        
        return logits, loss

# ══════════════════════════════════════════════════════════════════
# Observer setup with all features
# ══════════════════════════════════════════════════════════════════

config = ObserverConfig(
    # Enable everything for comprehensive monitoring
    track_profiler=True,
    profile_every_n_steps=100,  # Profile every 100 steps
    
    track_memory=True,
    track_throughput=True,
    track_loss=True,
    
    track_layer_graph=True,    # For visualization
    track_layer_health=True,   # Monitor for dead neurons
    track_sustainability=True, # Track efficiency
    
    track_carbon_emissions=True,
    carbon_country_iso="USA",
    
    track_console_logs=True,
    track_error_logs=True,
)

observer = Observer(
    project_id=2,
    config=config,
    run_name="mini_gpt_shakespeare",
)

# ══════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════

hyperparams = {
    "vocab_size": 65,       # Character-level
    "n_embd": 128,          # Embedding dimension
    "n_head": 4,            # Attention heads
    "n_layer": 4,           # Transformer blocks
    "block_size": 64,       # Context window
    "batch_size": 32,
    "learning_rate": 3e-4,
    "max_steps": 5000,
}

observer.log_hyperparameters(hyperparams)

# ══════════════════════════════════════════════════════════════════
# Create and register model
# ══════════════════════════════════════════════════════════════════

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniGPT(
    vocab_size=hyperparams["vocab_size"],
    n_embd=hyperparams["n_embd"],
    n_head=hyperparams["n_head"],
    n_layer=hyperparams["n_layer"],
    block_size=hyperparams["block_size"],
).to(device)

observer.register_model(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams["learning_rate"])

# ══════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════

batch_size = hyperparams["batch_size"]
block_size = hyperparams["block_size"]

for step in range(hyperparams["max_steps"]):
    # Generate random training data (replace with real data)
    x = torch.randint(0, hyperparams["vocab_size"], (batch_size, block_size)).to(device)
    y = torch.randint(0, hyperparams["vocab_size"], (batch_size, block_size)).to(device)
    
    # Profiling or regular step
    if observer.should_profile(step):
        logits, loss = observer.profile_step(model, x, y)
    else:
        logits, loss = model(x, y)
        loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Record step
    observer.step(
        step=step,
        loss=loss,
        batch_size=batch_size,
        seq_length=block_size,  # Important for token throughput!
    )
    
    # Flush every 100 steps (or every epoch)
    if (step + 1) % 100 == 0:
        record = observer.flush()
        print(f"Step {step + 1}: loss={record['loss']['train_mean']:.4f}, "
              f"tokens/sec={record['throughput']['tokens_per_second']:.0f}")

# ══════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════

observer.export()  # Uses default path
observer.close()
```

### Example 3: Using Context Manager (Cleaner Code)

```python
from observer import Observer, ObserverConfig

config = ObserverConfig(track_profiler=True)

# Using 'with' automatically calls close() at the end
with Observer(project_id=1, config=config, run_name="clean_example") as obs:
    obs.log_hyperparameters({"lr": 0.001})
    obs.register_model(model)
    
    for epoch in range(10):
        for step, (x, y) in enumerate(dataloader):
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            obs.step(step, loss, batch_size=x.size(0))
        
        obs.flush()
    
    obs.export("my_report.json")
# obs.close() is called automatically here!
```

---

## 11. Troubleshooting

### Common Errors and Solutions

#### 1. "Failed to create backend session"

```
[Observer] WARNING: Failed to create backend session: <urlopen error [Errno 111] Connection refused>
```

**Problem:** The backend server isn't running.

**Solution:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

#### 2. "Project not found"

```
HTTPError: 404 Not Found
```

**Problem:** The project_id doesn't exist in the database.

**Solution:** Create the project first via the API or use an existing project ID.

```bash
# Create a project via curl:
curl -X POST "http://localhost:8000/projects" \
  -H "Content-Type: application/json" \
  -d '{"name": "My Project"}'
```

#### 3. "Session is not running"

```
HTTPError: 400 Bad Request - Session is not running
```

**Problem:** You tried to register a step after the session was stopped/paused.

**Solution:** Check the session status. If pending, either resume or create a new session.

#### 4. "codecarbon not installed"

```
[Observer] WARNING: codecarbon not installed; carbon tracking disabled. pip install codecarbon
```

**Problem:** Carbon tracking requires the codecarbon package.

**Solution:**
```bash
pip install codecarbon
```
Or disable carbon tracking:
```python
config = ObserverConfig(track_carbon_emissions=False)
```

#### 5. Memory Issues During Profiling

**Problem:** Profiling uses extra memory and may cause OOM errors.

**Solution:**
```python
config = ObserverConfig(
    track_profiler=True,
    profile_every_n_steps=None,  # Disable periodic profiling
    profile_at_step=0,           # Only profile first step
    profiler_profile_memory=False,  # Don't profile memory allocations
)
```

### Debugging Tips

1. **Enable debug logging:**
   ```python
   import logging
   config = ObserverConfig(log_level=logging.DEBUG)
   ```

2. **Check what's being sent to backend:**
   ```python
   # After flush(), look at the returned record
   record = observer.flush()
   print(record.keys())  # What data was collected
   print(record['loss'])  # Loss statistics
   ```

3. **Verify backend is receiving data:**
   ```bash
   # Check the database directly
   sqlite3 backend/app.db "SELECT * FROM trainstep ORDER BY id DESC LIMIT 5;"
   ```

4. **Test backend connection:**
   ```python
   import requests
   response = requests.get("http://localhost:8000/projects")
   print(response.json())
   ```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     OBSERVER QUICK REFERENCE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CREATION:                                                               │
│    obs = Observer(project_id=1, config=ObserverConfig(), run_name="...")│
│                                                                          │
│  SETUP:                                                                  │
│    obs.log_hyperparameters({"lr": 0.001, ...})                          │
│    obs.register_model(model)                                            │
│                                                                          │
│  DURING TRAINING:                                                        │
│    if obs.should_profile(step):                                         │
│        logits, loss = obs.profile_step(model, x, y)                     │
│    else:                                                                 │
│        logits, loss = model(x, y); loss.backward()                      │
│    obs.step(step, loss, batch_size=B, seq_length=T)                     │
│                                                                          │
│  END OF EPOCH:                                                           │
│    record = obs.flush(val_metrics={"val_loss": v})                      │
│                                                                          │
│  CLEANUP:                                                                │
│    obs.export("path/to/report.json")                                    │
│    obs.close()                                                          │
│                                                                          │
│  BACKEND ENDPOINTS:                                                      │
│    POST /sessions/project/{id}  → Create session                        │
│    POST /sessions/{id}/model    → Register model                        │
│    POST /sessions/{id}/step     → Register step (triggers diagnostics)  │
│    POST /sessions/{id}/log      → Push log message                      │
│    GET  /sessions/{id}/status   → Check status                          │
│    POST /sessions/{id}/action   → Stop/resume                           │
│                                                                          │
│  SESSION STATUSES:                                                       │
│    running → analyzing → running (loop)                                 │
│    running → analyzing → pending (issues found!)                        │
│    pending → stopped | running (user choice)                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

1. **Observer** is a Python class that monitors PyTorch training
2. It collects: loss, throughput, memory, profiles, layer health, carbon emissions
3. It communicates with a **FastAPI backend** via HTTP requests
4. The backend stores everything in a **SQLite database**
5. After each step, the backend runs **diagnostics** to find problems
6. If critical issues are found, training **pauses** (status = "pending")
7. Everything is designed to help you train better, faster, greener models!

Good luck with your ML journey! 🚀
