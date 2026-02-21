# ML Diagnostics Engine — Complete Guide for Beginners

> **What is this?** This is a "Sentry for ML Pipelines" — an automated system that analyzes your machine learning training runs and tells you what's going wrong, why, and how to fix it.

---

## Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [How Does It Work? (The Big Picture)](#2-how-does-it-work-the-big-picture)
3. [Core Concepts](#3-core-concepts)
4. [The Diagnostics Engine Explained](#4-the-diagnostics-engine-explained)
5. [All Diagnostic Checks (With Examples)](#5-all-diagnostic-checks-with-examples)
6. [GreenAI / Sustainability Checks](#6-greenai--sustainability-checks)
7. [Architecture-Specific Checks (CNN)](#7-architecture-specific-checks-cnn)
8. [API Endpoints](#8-api-endpoints)
9. [Understanding the Output](#9-understanding-the-output)
10. [Code Examples](#10-code-examples)
11. [Future Improvements](#11-future-improvements)

---

## 1. What Problem Does This Solve?

### The Problem

When you train a neural network, many things can go wrong:

- **Your loss might explode** (go from 0.5 to infinity) 💥
- **Your model might overfit** (gets great on training data, terrible on test data)
- **Your training might be wasting energy** (training for 100 epochs when 20 would be enough)
- **Layers might be "dead"** (not learning anything)
- **You might be using 97% of parameters on one layer** (very inefficient)

**Without this system:** You'd have to manually check logs, graphs, and metrics to find these problems.

**With this system:** It automatically detects 30+ types of problems and tells you exactly what's wrong and how to fix it!

---

## 2. How Does It Work? (The Big Picture)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        YOUR TRAINING CODE                                │
│  for epoch in range(100):                                               │
│      for batch in dataloader:                                           │
│          loss = model(batch)                                            │
│          loss.backward()                                                │
│          optimizer.step()                                               │
│          observer.step(...)   ◄─── Collects metrics every step         │
│      observer.flush()         ◄─── Saves epoch data                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        OBSERVER (neural_network/observer.py)             │
│  Collects:                                                              │
│    • Loss values (mean, std, min, max)                                  │
│    • Throughput (samples per second)                                    │
│    • Memory usage (RAM, GPU memory)                                     │
│    • Profiler data (which layers take how long)                         │
│    • Layer health (gradients, activations, weights)                     │
│    • Carbon emissions (CO2 and energy usage)                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BACKEND DATABASE                                  │
│  Tables:                                                                │
│    • Project       (your project info)                                  │
│    • TrainSession  (one training run)                                   │
│    • TrainStep     (one epoch's data)                                   │
│    • Model         (architecture info)                                  │
│    • SessionLog    (error/console logs)                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              DIAGNOSTICS ENGINE (backend/diagnostics/engine.py)          │
│                                                                         │
│  Runs 30+ heuristic checks:                                             │
│    ✓ Loss problems (explosion, plateau, divergence, overfitting)        │
│    ✓ System problems (memory leaks, slow epochs, high CPU)              │
│    ✓ Profiler problems (compute hotspots, backward dominance)           │
│    ✓ Sustainability (wasted compute, dead layers, vanishing gradients)  │
│    ✓ Architecture (for CNNs - missing pooling, large kernels, etc.)     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT                                            │
│  {                                                                       │
│    "health_score": 73,                                                  │
│    "issues": [                                                          │
│      {                                                                  │
│        "severity": "warning",                                           │
│        "title": "Overfitting detected",                                 │
│        "description": "Validation loss rising while training falls...", │
│        "suggestion": "Add dropout, weight decay, or data augmentation"  │
│      }                                                                  │
│    ]                                                                    │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Concepts

### 3.1 Issues

An **issue** is a problem the diagnostics engine found in your training. Every issue has:

```python
@dataclass
class IssueData:
    severity: IssueSeverity     # How bad is this? (critical/warning/info)
    category: IssueCategory     # What type of problem?
    title: str                  # Short name like "Loss explosion"
    description: str            # Detailed explanation of what's happening
    epoch_index: int | None     # Which epoch did this happen in?
    layer_id: str | None        # Which layer is affected?
    metric_key: str | None      # Which metric flagged this?
    metric_value: Any           # The actual value that triggered the issue
    suggestion: str             # How to fix this problem!
```

### 3.2 Severity Levels

| Severity | Weight | Meaning |
|----------|--------|---------|
| `critical` | -20 points | **Stop and fix this NOW!** Training is broken |
| `warning` | -7 points | **Should fix soon.** Hurting your results |
| `info` | -2 points | **Good to know.** Potential optimization |

### 3.3 Categories

```python
class IssueCategory(str, Enum):
    loss = "loss"                      # Problems with your loss function
    throughput = "throughput"          # Training speed problems
    memory = "memory"                  # Memory leaks or usage issues
    profiler = "profiler"              # Compute bottlenecks
    logs = "logs"                      # Error logs detected
    system = "system"                  # CPU/system issues
    architecture = "architecture"      # Model structure problems
    sustainability = "sustainability"  # Wasted compute/energy (GreenAI)
```

### 3.4 Health Score

The **health score** is a number from 0 to 100:

```python
def compute_health_score(issues: list[IssueData]) -> int:
    score = 100
    for issue in issues:
        if issue.severity == "critical":
            score -= 20
        elif issue.severity == "warning":
            score -= 7
        elif issue.severity == "info":
            score -= 2
    return max(0, score)  # Never go below 0
```

**Example:**
- No issues → Score: 100 ✅
- 1 critical + 2 warnings → Score: 100 - 20 - 7 - 7 = 66 ⚠️
- 5 criticals → Score: 0 ❌

---

## 4. The Diagnostics Engine Explained

### 4.1 What Data Does It Analyze?

The engine analyzes three types of data:

#### Epochs (from TrainStep.payload)

Each epoch contains:

```python
epoch_data = {
    "epoch": 0,                          # Which epoch number
    "duration_seconds": 12.5,            # How long it took
    
    # Loss data
    "loss": {
        "train_mean": 2.3456,            # Average training loss
        "train_std": 0.15,               # Standard deviation (how spread out)
        "train_min": 2.1,                # Lowest batch loss
        "train_max": 2.8,                # Highest batch loss
        "val": {
            "val_loss": 2.5,             # Validation loss
            "val_acc": 0.85              # Validation accuracy
        }
    },
    
    # How fast training is going
    "throughput": {
        "samples_per_second": 1200,      # Batches processed per second
        "samples_processed": 50000       # Total samples this epoch
    },
    
    # Memory usage
    "memory": {
        "process_rss_mb": 2048,          # RAM used in megabytes
    },
    
    # CPU/system info
    "system": {
        "cpu_percent": 85                # How much CPU is being used
    },
    
    # Profiler data (which layers are slow)
    "profiler": {
        "total_cpu_time_ms": 1500,       # Total CPU time
        "total_cuda_time_ms": 0,         # Total GPU time
        "fwd_bwd_ratio": 0.3,            # Forward/Backward ratio
        "per_layer": [                   # Per-layer breakdown
            {"name": "conv1", "pct_total": 20.5},
            {"name": "fc1", "pct_total": 35.2}
        ],
        "operation_categories": {
            "forward_pass": {"pct_cpu": 30},
            "backward_pass": {"pct_cpu": 45}
        }
    },
    
    # Layer health (new GreenAI feature!)
    "layer_health": {
        "layers": {
            "conv1": {
                "gradient_norm_mean": 0.0001,
                "has_vanishing_gradients": True,
                "weight_sparsity": 0.2,
                "activation_var_of_means": 0.001
            }
        }
    },
    
    # Carbon emissions (CodeCarbon integration)
    "carbon_emissions": {
        "epoch_co2_kg": 0.000023,        # CO2 emitted in kg
        "epoch_energy_kwh": 0.0001,      # Energy used in kWh
        "power_draw_watts": 45           # Power consumption
    }
}
```

#### Logs (SessionLog objects)

```python
log_entry = {
    "ts": "2026-02-21T10:30:15",
    "level": "ERROR",
    "msg": "CUDA out of memory",
    "kind": "error"  # or "console"
}
```

#### Architecture (Model.architecture)

```python
architecture = {
    "total_parameters": 1_200_000,
    "layers": {
        "conv1": {
            "type": "Conv2d",
            "parameters": 320,
            "pct_of_total": 0.05
        },
        "fc1": {
            "type": "Linear",
            "parameters": 1_100_000,
            "pct_of_total": 92.0
        }
    },
    "layer_graph": {
        "nodes": [
            {"id": "conv1", "type": "Conv2d", "category": "convolution", 
             "in_channels": 1, "out_channels": 32, "kernel_size": [5, 5]},
            {"id": "fc1", "type": "Linear", "category": "linear"}
        ],
        "sequential_path": ["conv1", "pool1", "conv2", "fc1"]
    }
}
```

---

## 5. All Diagnostic Checks (With Examples)

### Category 1: Loss Checks

#### 5.1 `check_loss_divergence` — Loss Going Up

**What it detects:** Your loss is increasing for 2 or more epochs in a row (bad sign!)

**Example scenario:**
```
Epoch 0: loss = 2.5
Epoch 1: loss = 2.7  (went up)
Epoch 2: loss = 3.0  (went up again!) ⚠️ DIVERGENCE DETECTED
```

**Code:**
```python
def check_loss_divergence(epochs: list[dict]) -> list[IssueData]:
    """Detect 2+ consecutive epochs where train_mean is rising after epoch 0."""
    issues = []
    if len(epochs) < 3:
        return issues
    
    for i in range(2, len(epochs)):
        prev_prev = epochs[i - 2]["loss"]["train_mean"]  # 2 epochs ago
        prev = epochs[i - 1]["loss"]["train_mean"]       # 1 epoch ago
        curr = epochs[i]["loss"]["train_mean"]           # current
        
        # If loss went up twice in a row...
        if curr > prev > prev_prev:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                category=IssueCategory.loss,
                title="Loss divergence detected",
                suggestion="Check your learning rate — try a scheduler"
            ))
            break  # Only report first occurrence
    return issues
```

**How to fix:**
- Reduce learning rate
- Use a learning rate scheduler (like `CosineAnnealingLR`)
- Check for NaN values in your data

---

#### 5.2 `check_loss_explosion` — Loss Doubled!

**What it detects:** Loss more than doubled between epochs (very bad!)

**Example scenario:**
```
Epoch 0: loss = 0.5
Epoch 1: loss = 1.5  (3x increase!) 🔴 CRITICAL
```

**Code:**
```python
def check_loss_explosion(epochs: list[dict]) -> list[IssueData]:
    """Detect a >100% jump in train_mean between consecutive epochs."""
    issues = []
    for i in range(1, len(epochs)):
        prev = epochs[i - 1]["loss"]["train_mean"]
        curr = epochs[i]["loss"]["train_mean"]
        
        if prev > 0 and curr / prev > 2.0:  # More than doubled
            issues.append(IssueData(
                severity=IssueSeverity.critical,  # CRITICAL!
                category=IssueCategory.loss,
                title="Loss explosion",
                description=f"Loss went from {prev:.4f} to {curr:.4f}",
                suggestion="Lower learning rate + add gradient clipping"
            ))
    return issues
```

**How to fix:**
```python
# 1. Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 2. Lower your learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # was 1e-3
```

---

#### 5.3 `check_loss_plateau` — Loss Stopped Improving

**What it detects:** Less than 1% improvement over the last 3 epochs

**Example scenario:**
```
Epoch 5: loss = 0.500
Epoch 6: loss = 0.498  (0.4% improvement)
Epoch 7: loss = 0.497  (0.2% improvement) ⚠️ PLATEAU
```

**How to fix:**
```python
# Use a learning rate scheduler that adjusts when loss plateaus
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

# After each epoch:
scheduler.step(val_loss)
```

---

#### 5.4 `check_overfitting` — Model Memorizing, Not Learning

**What it detects:** Training loss going down but validation loss going up

**Example scenario:**
```
         Training    Validation
Epoch 1:  2.5         2.6
Epoch 2:  2.3         2.7  ← val went UP while train went DOWN
Epoch 3:  2.1         3.0  ← happening again! ⚠️ OVERFITTING
```

**Code:**
```python
def check_overfitting(epochs: list[dict]) -> list[IssueData]:
    overfit_epochs = []
    for i in range(1, len(epochs)):
        val_prev = epochs[i - 1]["loss"]["val"]["val_loss"]
        val_curr = epochs[i]["loss"]["val"]["val_loss"]
        train_prev = epochs[i - 1]["loss"]["train_mean"]
        train_curr = epochs[i]["loss"]["train_mean"]
        
        # Overfitting: val goes UP while train goes DOWN
        if val_curr > val_prev and train_curr < train_prev:
            overfit_epochs.append(i)
    
    if len(overfit_epochs) >= 2:  # Happened twice = real problem
        return [IssueData(...)]
    return []
```

**How to fix:**
```python
# 1. Add Dropout
class MyModel(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout
        return self.fc2(x)

# 2. Use weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 3. Add data augmentation
from torchvision import transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])
```

---

#### 5.5 `check_high_loss_variance` — Noisy Training

**What it detects:** Loss varies wildly within an epoch (std/mean > 50%)

**Example scenario:**
```
Epoch 3: mean_loss = 1.0, std_loss = 0.6  (60% variance!) ⚠️
Different batches giving wildly different losses
```

**How to fix:**
```python
# Increase batch size (more stable gradients)
dataloader = DataLoader(dataset, batch_size=128)  # was 32

# Or lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # was 1e-3
```

---

#### 5.6 `check_gradient_instability` — Gradient Spikes

**What it detects:** One batch had a loss 20x higher than the mean

**Example scenario:**
```
Epoch 2: mean_loss = 0.5, max_loss = 15.0  (30x the mean!) ⚠️
One batch caused a massive gradient spike
```

**How to fix:**
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

### Category 2: System/Throughput Checks

#### 5.7 `check_throughput_degradation` — Training Slowing Down

**What it detects:** Speed dropped more than 20% below the peak

**Example scenario:**
```
Epoch 0: 1000 samples/sec (peak)
Epoch 5: 750 samples/sec ⚠️ (25% slower than peak!)
```

**How to fix:**
- Check if another process is using CPU/GPU
- Increase `num_workers` in DataLoader
- Check for I/O bottlenecks (slow disk)

---

#### 5.8 `check_memory_growth` — Memory Leak!

**What it detects:** RAM usage grew by more than 25% during training

**Example scenario:**
```
Epoch 0: 2000 MB RAM
Epoch 10: 3000 MB RAM ⚠️ (50% growth - memory leak!)
```

**How to fix:**
```python
# WRONG - keeps tensors in memory
loss_history = []
for batch in dataloader:
    loss = model(batch)
    loss_history.append(loss)  # BAD! Holds tensor

# RIGHT - only keep the number
loss_history = []
for batch in dataloader:
    loss = model(batch)
    loss_history.append(loss.item())  # GOOD! Just the number
```

---

#### 5.9 `check_slow_epoch` — One Epoch Was Unusually Slow

**What it detects:** An epoch took 1.5x longer than the median

**Example scenario:**
```
Epochs 0-9: ~10 seconds each (median)
Epoch 7: 18 seconds ⚠️ (1.8x longer!)
```

---

#### 5.10 `check_high_cpu` — CPU Bottleneck

**What it detects:** CPU usage over 90%

**Suggestion:** Move to GPU!

```python
# Move model and data to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# In training loop:
x, y = x.to(device), y.to(device)
```

---

#### 5.11 `check_error_logs` — Errors Detected!

**What it detects:** Any error-level log entries

**Example:**
```
Found 3 error logs!
First error: "CUDA out of memory. Tried to allocate 2.00 GiB"
```

---

### Category 3: Profiler Checks

#### 5.12 `check_profiler_hotspot` — One Layer Uses 40%+ of Compute

**What it detects:** A single layer is bottlenecking your training

**Example scenario:**
```
conv1: 15% of compute
fc1: 45% of compute ⚠️ HOTSPOT
fc2: 10% of compute
```

**How to fix:**
- Make that layer smaller
- Use more efficient operations
- Restructure the network

---

#### 5.13 `check_backward_dominance` — Backward Pass Too Heavy

**What it detects:** More than 45% of time spent on backward pass

**Normal:** Forward ~40%, Backward ~40%, Other ~20%
**Problem:** Forward ~25%, Backward ~60%, Other ~15%

**How to fix:**
```python
# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Saves memory by recomputing during backward
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

---

#### 5.14 `check_fwd_bwd_ratio` — Extremely Heavy Backward

**What it detects:** Forward/backward ratio below 0.15 (backward 6x heavier)

---

## 6. GreenAI / Sustainability Checks

These checks help you save energy and compute! 🌱

### 6.1 `check_diminishing_returns` — Training Not Worth It

**What it detects:** Marginal improvement is less than 5% of total improvement

**Example scenario:**
```
Epoch 0: loss = 2.0
Epoch 5: loss = 1.0  (cumulative improvement: 1.0)
Epoch 6: loss = 0.98 (marginal improvement: 0.02)

Marginal/Cumulative = 0.02/1.0 = 2% ⚠️ DIMINISHING RETURNS
You improved only 2% this epoch but used just as much compute!
```

**How to fix:** Implement early stopping!

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,        # Wait 3 epochs before stopping
    min_delta=0.01     # Minimum improvement to count as progress
)
```

---

### 6.2 `check_early_stop_opportunity` — Wasted Compute Detected

**What it detects:** Training could have stopped earlier

**Example scenario:**
```
Epoch 0-5: Good progress
Epoch 6: Diminishing returns start
Epoch 7-10: Minimal improvement

RESULT: Could have stopped at epoch 5!
Wasted: 5 epochs × 60 seconds = 300 seconds of compute
```

---

### 6.3 `check_device_underutilization` — Not Using GPU!

**What it detects:** Training entirely on CPU when GPU available

**Output:**
```
"All 10 profiled epochs show 0ms CUDA time and 5000ms CPU time.
GPU-accelerated training is typically 10-100x faster!"
```

**How to fix:**
```python
model = model.to('cuda')
x = x.to('cuda')
y = y.to('cuda')
```

---

### 6.4 `check_over_parameterized_layer` — Too Many Unused Params

**What it detects:** Layer has 10x more params than its compute share

**Example scenario:**
```
fc1: Has 50% of parameters but only 5% of compute
fc1 param_to_compute_ratio = 10x ⚠️ OVER-PARAMETERIZED
These parameters aren't doing much work!
```

---

### 6.5 `check_compute_inefficient_layer` — Layer Uses Too Much Compute

**What it detects:** Layer uses 10x more compute than its param share

**Example scenario:**
```
conv1: Has 0.1% of parameters but 20% of compute
conv1 compute_to_param_ratio = 200x ⚠️
Consider depthwise separable convolutions
```

---

### 6.6 `check_dead_neurons` — Layer Not Learning

**What it detects:** Layer has >50% near-zero weights or is completely dead

**What "dead" means:**
- Output never changes between batches
- AND more than 90% of weights are near zero

**How to fix:**
```python
# Option 1: Reinitialize weights
nn.init.kaiming_normal_(layer.weight)

# Option 2: Remove the layer (it's not contributing)
# Option 3: Use different activation (ReLU → LeakyReLU)
```

---

### 6.7 `check_vanishing_gradients` — Gradients Disappearing

**What it detects:** Gradient norm extremely small (< 1e-7)

**Why it matters:** If gradients are near zero, the layer can't learn!

**Example output:**
```
Layer 'conv1' has near-zero gradient flow in 3 epochs.
Average gradient norm: 2.5e-8
```

**How to fix:**
```python
# 1. Add skip connections (ResNet style)
class ResBlock(nn.Module):
    def forward(self, x):
        return x + self.layers(x)  # Skip connection!

# 2. Use better initialization
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# 3. Try different activations
activation = nn.GELU()  # Instead of ReLU
```

---

### 6.8 `check_frozen_output` — Layer Output Never Changes

**What it detects:** Same output regardless of input

**This means:** The layer is NOT responding to different inputs!

---

### 6.9 `check_activation_collapse` — All Outputs Are The Same

**What it detects:** Near-zero activation variance

```
Layer 'fc1' outputs almost the same value for every input!
This is "representation collapse" - the layer lost its ability to distinguish.
```

**How to fix:**
```python
# Add normalization before the layer
self.norm = nn.BatchNorm1d(features)
# or
self.norm = nn.LayerNorm(features)
```

---

### 6.10 `check_redundant_layers` — Two Layers Doing The Same Thing

**What it detects:** Two consecutive layers with >95% correlation

**Example:**
```
Layer 'fc1' and 'fc2' produce nearly identical outputs!
Average correlation: 0.97
You might be able to remove one of them.
```

---

### 6.11 Carbon Footprint Checks

#### `check_high_carbon_intensity`

**What it detects:** Reports total CO2 emissions and flags wasteful epochs

**Example output:**
```
Training carbon footprint: 23.5g CO2
Total emissions: 23.5g CO2 over 10 epochs
Energy consumed: 150 Wh
Average power draw: 65W
```

#### `check_wasted_carbon`

**What it detects:** CO2 wasted on unnecessary epochs

**Example output:**
```
Training could have stopped after epoch 5.
The remaining 5 epochs wasted 12.3g CO2 and 75 Wh
with diminishing returns.
```

---

## 7. Architecture-Specific Checks (CNN)

These checks only run on Convolutional Neural Networks (detected automatically).

### 7.1 `_check_fc_dominates` — FC Layer Has Too Many Params

**What it detects:** A Linear layer holding >92% of all parameters

**Example scenario:**
```
conv1: 320 params (0.05%)
conv2: 4,800 params (0.8%)
fc1: 1,000,000 params (97%) ⚠️ TOO MANY!
```

**Why this is bad:** In a CNN, conv layers should do most of the work. If FC dominates, your feature maps are probably too large before flattening.

**How to fix:**
```python
# Add adaptive pooling BEFORE the FC layer
self.pool = nn.AdaptiveAvgPool2d(1)  # Collapses spatial dims to 1x1

# Then your FC will be much smaller:
# Instead of fc(32 * 28 * 28, 10) = 25,088 inputs
# You get fc(32, 10) = 32 inputs
```

---

### 7.2 `_check_conv_bottleneck` — One Conv Layer Has >60% Params

**What it detects:** Single conv layer is disproportionately large

**How to fix:**
```python
# Use bottleneck blocks (ResNet style)
# Instead of: Conv2d(256, 256, 3)
# Use:
nn.Sequential(
    nn.Conv2d(256, 64, 1),    # 1x1 to reduce channels
    nn.Conv2d(64, 64, 3),     # 3x3 at smaller dim
    nn.Conv2d(64, 256, 1),    # 1x1 to expand back
)
```

---

### 7.3 `_check_missing_pooling` — Too Many Convs Without Pooling

**What it detects:** 3 or more consecutive conv layers without pooling

**Why this matters:** Without pooling, spatial dimensions stay large, making FC layers huge!

**How to fix:**
```python
# Add pooling between conv blocks
self.conv1 = nn.Conv2d(1, 32, 3)
self.pool1 = nn.MaxPool2d(2)    # <-- Add this!
self.conv2 = nn.Conv2d(32, 64, 3)
self.pool2 = nn.MaxPool2d(2)    # <-- And this!
```

---

### 7.4 `_check_large_kernel` — Kernel Size ≥7 Detected

**What it detects:** Large kernels (7×7 or bigger) on small inputs

**When it's fine:** Large inputs like ImageNet (224×224)
**When it's bad:** Small inputs like MNIST (28×28)

**How to fix:** Use 3×3 kernels for small inputs

---

### 7.5 `_check_early_channel_explosion` — Too Many Channels Early

**What it detects:** First conv expands from 1 channel to >32

**Example:**
```
Input: 1 channel (grayscale)
First conv output: 64 channels ⚠️ Too much for small images!
```

**How to fix:**
```python
# Start smaller for MNIST-like data
self.conv1 = nn.Conv2d(1, 8, 3)   # 8 channels, not 64
self.conv2 = nn.Conv2d(8, 16, 3)  # Gradual increase
```

---

## 8. API Endpoints

### POST `/diagnostics/sessions/{session_id}/run`

**What it does:** Runs the full diagnostic analysis on a training session

**Returns:**
```json
{
  "id": 1,
  "session_id": 123,
  "created_at": "2026-02-21T10:30:00",
  "health_score": 73,
  "issue_count": 5,
  "arch_type": "cnn",
  "summary_json": {
    "severity_breakdown": {"critical": 0, "warning": 3, "info": 2},
    "category_breakdown": {"loss": 2, "sustainability": 3}
  },
  "issues": [...],
  "epoch_trends": [...],
  "session_level_issues": [...],
  "layer_highlights": [...],
  "sustainability": {
    "optimal_stop_epoch": 5,
    "wasted_epochs": 3,
    "wasted_compute_pct": 30.5,
    "dead_layers": ["fc3"],
    "total_co2_kg": 0.0023
  }
}
```

### GET `/diagnostics/sessions/{session_id}`

**What it does:** Lists all past diagnostic runs for a session (summary only)

### GET `/diagnostics/runs/{run_id}`

**What it does:** Gets full details for a specific diagnostic run

### GET `/diagnostics/sessions/{session_id}/health`

**What it does:** Quick health check (latest run or computed on-the-fly)

**Returns:**
```json
{
  "session_id": 123,
  "health_score": 73,
  "severity_counts": {"critical": 0, "warning": 3, "info": 2},
  "top_issues": [...],
  "top_layers": [...]
}
```

### GET `/diagnostics/projects/{project_id}/trend`

**What it does:** Shows improvement trend across all sessions in a project

---

## 9. Understanding the Output

### 9.1 DiagnosticRunOut Schema

```python
class DiagnosticRunOut(BaseModel):
    id: int                           # Unique run ID
    session_id: int                   # Which training session
    created_at: datetime              # When analysis ran
    health_score: int                 # 0-100, higher is better
    issue_count: int                  # Total issues found
    arch_type: str                    # "cnn", "transformer", "rnn", "generic"
    summary_json: dict                # Quick stats
    
    issues: list[IssueOut]            # All detected issues
    epoch_trends: list[...]           # Issues grouped by epoch
    session_level_issues: list[...]   # Issues not tied to specific epoch
    layer_highlights: list[...]       # Issues grouped by layer
    sustainability: SustainabilityInsight  # Green AI metrics
```

### 9.2 Layer Highlights

```python
class LayerHighlight(BaseModel):
    layer_id: str                     # e.g., "fc1"
    layer_type: str | None            # e.g., "Linear"
    severity_score: int               # Sum of issue weights
    issues: list[IssueOut]            # All issues for this layer
```

Sorted by severity score descending — most problematic layers first!

### 9.3 SustainabilityInsight

```python
class SustainabilityInsight(BaseModel):
    # Early stopping
    optimal_stop_epoch: int | None    # When training should have stopped
    wasted_epochs: int | None         # How many extra epochs ran
    wasted_compute_pct: float | None  # % of compute wasted
    wasted_duration_seconds: float | None
    
    # Training totals
    total_training_duration_seconds: float | None
    total_samples_processed: int | None
    
    # Layer health
    dead_layers: list[str]                    # ["fc3", "conv5"]
    vanishing_gradient_layers: list[str]      # ["conv1"]
    frozen_output_layers: list[str]           # []
    redundant_layer_pairs: list[dict]         # [{"layer_a": "fc1", "layer_b": "fc2"}]
    
    # Efficiency score (0-100)
    parameter_efficiency_score: float | None
    
    # Carbon footprint
    total_co2_kg: float | None
    total_energy_kwh: float | None
    co2_per_epoch_avg_kg: float | None
    co2_per_1k_samples_kg: float | None
    avg_power_draw_watts: float | None
    wasted_co2_kg: float | None
```

---

## 10. Code Examples

### 10.1 Using the Observer in Your Training

```python
from observer import Observer, ObserverConfig

# Create observer with all tracking enabled
config = ObserverConfig(
    track_profiler=True,
    track_layer_health=True,
    track_carbon_emissions=True,
    profile_at_step=0  # Profile first step of each epoch
)

obs = Observer(
    project_id=1,
    config=config,
    run_name="my_experiment_v1"
)

# Register your model
obs.register_model(model)

# Log hyperparameters
obs.log_hyperparameters({
    "learning_rate": 1e-3,
    "batch_size": 64,
    "optimizer": "Adam"
})

# Training loop
for epoch in range(10):
    for step, (x, y) in enumerate(train_loader):
        # Use profiled step for the first batch
        if obs.should_profile(step):
            logits, loss = obs.profile_step(model, x, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Record the step
        obs.step(step, loss, batch_size=x.size(0))
    
    # End of epoch - save metrics
    val_metrics = {"val_loss": val_loss, "val_acc": val_acc}
    obs.flush(val_metrics=val_metrics)

# Save report
obs.export("observer_reports/my_run.json")
obs.close()
```

### 10.2 Running Diagnostics via API (Python)

```python
import requests

# Run diagnostics on session 123
response = requests.post(
    "http://localhost:8000/diagnostics/sessions/123/run"
)
result = response.json()

print(f"Health Score: {result['health_score']}/100")
print(f"Issues Found: {result['issue_count']}")

# Print critical issues
for issue in result['issues']:
    if issue['severity'] == 'critical':
        print(f"🔴 CRITICAL: {issue['title']}")
        print(f"   {issue['description']}")
        print(f"   Fix: {issue['suggestion']}")
        print()

# Print sustainability insights
sus = result['sustainability']
if sus['wasted_epochs']:
    print(f"⚠️ You could have stopped {sus['wasted_epochs']} epochs earlier!")
    print(f"   Wasted compute: {sus['wasted_compute_pct']}%")
    if sus['wasted_co2_kg']:
        print(f"   Wasted CO2: {sus['wasted_co2_kg'] * 1000:.2f}g")
```

### 10.3 Running Diagnostics via cURL

```bash
# Run diagnostics
curl -X POST http://localhost:8000/diagnostics/sessions/123/run

# Get health summary
curl http://localhost:8000/diagnostics/sessions/123/health

# Get project trend
curl http://localhost:8000/diagnostics/projects/1/trend
```

### 10.4 Interpreting Results in Your Code

```python
def should_stop_early(result):
    """Use diagnostics to decide if training should stop."""
    # Health score too low
    if result['health_score'] < 50:
        print("Health score critical - investigate issues")
        return True
    
    # Check for critical issues
    criticals = [i for i in result['issues'] if i['severity'] == 'critical']
    if criticals:
        print(f"Found {len(criticals)} critical issues!")
        return True
    
    # Check diminishing returns
    sus = result['sustainability']
    if sus.get('wasted_compute_pct', 0) > 20:
        print("Training past optimal point - consider stopping")
        return True
    
    return False

def fix_issues(result):
    """Provide automated suggestions based on issues."""
    for issue in result['issues']:
        if issue['title'] == 'Loss explosion':
            print("SUGGESTION: Add gradient clipping!")
            print("  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)")
        
        elif issue['title'] == 'Overfitting detected':
            print("SUGGESTION: Add regularization!")
            print("  - Use nn.Dropout(0.5)")
            print("  - Use AdamW with weight_decay=0.01")
        
        elif 'Dead layer' in issue['title']:
            layer = issue['layer_id']
            print(f"SUGGESTION: Layer '{layer}' is dead!")
            print(f"  - Reinitialize: nn.init.kaiming_normal_(model.{layer}.weight)")
```

---

## 11. Future Improvements

This section outlines how we can extend and improve the ML Diagnostics Engine.

### 11.1 New Architecture Checkers

Currently, we have robust checks for **CNNs**, but **Transformers** and **RNNs** have placeholder checkers. Here's what we should add:

#### Transformer-Specific Checks

```python
class TransformerChecker:
    """Transformer-specific architecture checks."""
    
    def run(self, arch: dict) -> list[IssueData]:
        issues = []
        issues += self._check_attention_head_count(arch)
        issues += self._check_embedding_dimension(arch)
        issues += self._check_position_encoding(arch)
        issues += self._check_layer_norm_placement(arch)
        issues += self._check_attention_entropy(arch)  # Already have data!
        return issues
    
    def _check_attention_head_count(self, arch):
        """Flag when n_head doesn't divide n_embd evenly."""
        # E.g., n_embd=128, n_head=3 = BAD (42.67 per head)
        # n_embd=128, n_head=4 = GOOD (32 per head)
        pass
    
    def _check_embedding_dimension(self, arch):
        """Flag oversized embeddings for small vocabulary."""
        # If vocab=1000 but n_embd=1024, that's probably overkill
        pass
    
    def _check_position_encoding(self, arch):
        """Warn if block_size exceeds position encoding capacity."""
        pass
    
    def _check_attention_entropy(self, arch):
        """Flag attention heads with very low entropy (always attending same position)."""
        # We already collect attention_entropy in the observer!
        pass
```

#### RNN-Specific Checks

```python
class RnnChecker:
    """RNN/LSTM/GRU-specific checks."""
    
    def run(self, arch: dict) -> list[IssueData]:
        issues = []
        issues += self._check_hidden_size_ratio(arch)
        issues += self._check_num_layers(arch)
        issues += self._check_bidirectional_usage(arch)
        issues += self._check_sequence_length(arch)
        return issues
    
    def _check_hidden_size_ratio(self, arch):
        """Flag when hidden_size >> input_size (overparameterized)."""
        pass
    
    def _check_num_layers(self, arch):
        """Warn when RNN has >4 layers (vanishing gradients risk)."""
        pass
```

### 11.2 New Diagnostic Checks to Implement

#### Learning Rate Checks

```python
def check_learning_rate_too_high(epochs: list[dict]) -> list[IssueData]:
    """Detect if learning rate is causing instability.
    
    Signs: Loss oscillating (going up and down repeatedly)
    """
    pass

def check_learning_rate_too_low(epochs: list[dict]) -> list[IssueData]:
    """Detect if learning rate is too conservative.
    
    Signs: Very slow but steady progress, could be faster
    """
    pass
```

#### Batch Size Checks

```python
def check_batch_size_too_small(epochs: list[dict]) -> list[IssueData]:
    """Detect if batch size is causing noisy gradients.
    
    Look at: throughput (GPU underutilized), loss variance
    """
    pass

def check_batch_size_too_large(epochs: list[dict]) -> list[IssueData]:
    """Detect if batch size is hurting generalization.
    
    Signs: Training loss very smooth but validation not improving
    """
    pass
```

#### Data Quality Checks

```python
def check_class_imbalance(epochs: list[dict]) -> list[IssueData]:
    """Detect if model is ignoring minority classes.
    
    Would need: per-class metrics in observer
    """
    pass

def check_data_leakage(epochs: list[dict]) -> list[IssueData]:
    """Detect if training and validation sets overlap.
    
    Signs: Validation accuracy suspiciously close to training
    """
    pass
```

#### Numerical Stability Checks

```python
def check_nan_in_loss(epochs: list[dict]) -> list[IssueData]:
    """Detect NaN values in loss history."""
    pass

def check_inf_gradients(epochs: list[dict]) -> list[IssueData]:
    """Detect infinite gradients."""
    # Would need gradient tracking in observer
    pass
```

### 11.3 Enhanced Observer Data Collection

To enable the new checks above, we'd need to collect more data:

```python
@dataclass
class ObserverConfig:
    # ... existing config ...
    
    # NEW: Per-class metrics
    track_per_class_metrics: bool = False  # accuracy/loss per class
    
    # NEW: Learning rate tracking
    track_learning_rate: bool = True  # log LR at each step
    
    # NEW: Gradient histogram
    track_gradient_histogram: bool = False  # more detailed gradient stats
    
    # NEW: Attention patterns (for transformers)
    track_attention_patterns: bool = False  # which tokens attend to which
    
    # NEW: Activation histograms
    track_activation_histogram: bool = False  # distribution of activations
```

### 11.4 Real-Time Diagnostics (WebSocket)

Currently, diagnostics run **after** training. We could add real-time alerts:

```python
# pseudocode for real-time diagnostics

class RealtimeDiagnostics:
    """Run checks as training progresses."""
    
    def __init__(self, websocket):
        self.ws = websocket
        self.epochs = []
    
    def on_epoch_end(self, epoch_data):
        self.epochs.append(epoch_data)
        
        # Run quick checks
        issues = []
        issues += check_loss_explosion(self.epochs[-2:])
        issues += check_loss_divergence(self.epochs[-3:])
        
        # Alert immediately if critical
        for issue in issues:
            if issue.severity == IssueSeverity.critical:
                self.ws.send({
                    "type": "alert",
                    "issue": issue
                })
```

### 11.5 ML-Based Anomaly Detection

Instead of just heuristics, use ML to detect unusual patterns:

```python
class MLAnomalyDetector:
    """Use machine learning to detect training anomalies."""
    
    def __init__(self, model_path: str):
        # Load pre-trained anomaly detection model
        self.model = load_model(model_path)
    
    def detect(self, epochs: list[dict]) -> list[IssueData]:
        # Convert epochs to feature matrix
        features = self._extract_features(epochs)
        
        # Run anomaly detection
        anomaly_scores = self.model.predict(features)
        
        # Convert high scores to issues
        issues = []
        for i, score in enumerate(anomaly_scores):
            if score > 0.9:
                issues.append(IssueData(
                    severity=IssueSeverity.warning,
                    category=IssueCategory.system,
                    title=f"Anomaly detected at epoch {i}",
                    description=f"This epoch's metrics are unusual (score: {score:.2f})",
                    suggestion="Review this epoch's logs and metrics manually."
                ))
        return issues
```

### 11.6 Automatic Fix Suggestions with Code Generation

Instead of just text suggestions, generate actual code fixes:

```python
def generate_fix_code(issue: IssueData, model_code: str) -> str:
    """Generate code to fix an issue automatically."""
    
    if issue.title == "Loss explosion":
        return f'''
# Add this after loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
'''
    
    elif issue.title == "Overfitting detected":
        # Parse model code and insert dropout
        return insert_dropout_into_model(model_code, rate=0.5)
    
    elif "Dead layer" in issue.title:
        layer = issue.layer_id
        return f'''
# Reinitialize the dead layer
nn.init.kaiming_normal_(model.{layer}.weight)
if model.{layer}.bias is not None:
    nn.init.zeros_(model.{layer}.bias)
'''
```

### 11.7 Comparison with Baseline

Track how your model compares to known baselines:

```python
class BaselineComparison:
    """Compare current run to historical baselines."""
    
    BASELINES = {
        "mnist_cnn": {"accuracy": 0.99, "epochs": 10, "params": 500_000},
        "cifar10_resnet": {"accuracy": 0.93, "epochs": 100, "params": 11_000_000},
    }
    
    def compare(self, session, task: str) -> list[IssueData]:
        if task not in self.BASELINES:
            return []
        
        baseline = self.BASELINES[task]
        issues = []
        
        # Compare accuracy
        if session.accuracy < baseline["accuracy"] * 0.9:
            issues.append(IssueData(
                severity=IssueSeverity.warning,
                title=f"Below baseline accuracy",
                description=f"Your accuracy ({session.accuracy:.2%}) is below "
                           f"the typical baseline ({baseline['accuracy']:.2%})"
            ))
        
        # Compare efficiency
        if session.params > baseline["params"] * 2:
            issues.append(IssueData(
                severity=IssueSeverity.info,
                title="Model larger than baseline",
                description=f"Your model has {session.params:,} params vs "
                           f"{baseline['params']:,} for typical {task} models"
            ))
        
        return issues
```

### 11.8 Integration Improvements

#### Weights & Biases Integration

```python
def export_to_wandb(diagnostic_run: DiagnosticRunOut):
    """Export diagnostics to Weights & Biases."""
    import wandb
    
    wandb.log({
        "health_score": diagnostic_run.health_score,
        "issue_count": diagnostic_run.issue_count,
        "critical_count": diagnostic_run.summary_json["severity_breakdown"]["critical"],
        "warning_count": diagnostic_run.summary_json["severity_breakdown"]["warning"],
    })
    
    # Log issues as a table
    issues_table = wandb.Table(columns=["severity", "title", "layer"])
    for issue in diagnostic_run.issues:
        issues_table.add_data(issue.severity, issue.title, issue.layer_id)
    wandb.log({"issues": issues_table})
```

#### Slack/Discord Notifications

```python
def send_slack_alert(issue: IssueData, webhook_url: str):
    """Send critical issues to Slack."""
    if issue.severity != IssueSeverity.critical:
        return
    
    payload = {
        "text": f"🔴 *Critical ML Issue Detected*\n"
               f"*{issue.title}*\n"
               f"{issue.description}\n"
               f"💡 {issue.suggestion}"
    }
    requests.post(webhook_url, json=payload)
```

### 11.9 UI/Dashboard Improvements

Ideas for the frontend visualization:

1. **Health Score Gauge** - Visual dial showing 0-100 score
2. **Issue Timeline** - Show when issues occurred during training
3. **Layer Heatmap** - Color-coded view of problematic layers
4. **Carbon Footprint Chart** - Track CO2 over time
5. **Comparison View** - Side-by-side runs to see which is better
6. **Suggested Actions Panel** - Prioritized list of fixes

### 11.10 Performance Optimizations

```python
# Current: Runs all checks sequentially
def run_diagnostics(epochs, logs, arch):
    issues = []
    issues += check_loss_divergence(epochs)
    issues += check_loss_explosion(epochs)
    # ... 28 more checks
    return issues

# Improved: Run checks in parallel
import concurrent.futures

def run_diagnostics_parallel(epochs, logs, arch):
    checks = [
        (check_loss_divergence, epochs),
        (check_loss_explosion, epochs),
        (check_overfitting, epochs),
        # ... group by data dependency
    ]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fn, data) for fn, data in checks]
        issues = []
        for future in concurrent.futures.as_completed(futures):
            issues.extend(future.result())
    
    return issues
```

### 11.11 Configurable Thresholds

Allow users to customize what triggers issues:

```python
@dataclass
class DiagnosticThresholds:
    """User-configurable thresholds for diagnostics."""
    
    # Loss checks
    loss_explosion_ratio: float = 2.0        # Default: 2x = explosion
    loss_plateau_improvement: float = 0.01   # Default: <1% = plateau
    overfitting_epochs: int = 2              # Default: 2 epochs = overfitting
    
    # Sustainability
    diminishing_returns_pct: float = 0.05    # Default: <5% marginal
    dead_neuron_sparsity: float = 0.5        # Default: >50% zeros = dead
    vanishing_gradient_threshold: float = 1e-7
    
    # System
    memory_growth_pct: float = 0.25          # Default: >25% growth
    throughput_drop_pct: float = 0.20        # Default: >20% drop

# Usage
thresholds = DiagnosticThresholds(
    loss_explosion_ratio=3.0,  # More lenient
    diminishing_returns_pct=0.10  # Stricter
)
run_diagnostics(epochs, logs, arch, thresholds=thresholds)
```

### 11.12 Testing & Validation

Add comprehensive tests for the diagnostic engine:

```python
# tests/test_diagnostics.py

def test_loss_explosion_detection():
    """Test that loss explosion is correctly detected."""
    epochs = [
        {"loss": {"train_mean": 1.0}},
        {"loss": {"train_mean": 5.0}},  # 5x increase
    ]
    issues = check_loss_explosion(epochs)
    
    assert len(issues) == 1
    assert issues[0].severity == IssueSeverity.critical
    assert "explosion" in issues[0].title.lower()

def test_no_false_positives_on_healthy_training():
    """Test that healthy training produces no critical issues."""
    # Generate healthy training data
    epochs = generate_healthy_training(n_epochs=10)
    
    issues, health_score, _ = run_diagnostics(epochs, [], None)
    
    critical_issues = [i for i in issues if i.severity == IssueSeverity.critical]
    assert len(critical_issues) == 0
    assert health_score >= 80

def test_all_checks_handle_empty_input():
    """Test that all checks gracefully handle empty input."""
    all_checks = [
        check_loss_divergence,
        check_loss_explosion,
        check_overfitting,
        # ... all other checks
    ]
    
    for check in all_checks:
        result = check([])  # Empty input
        assert isinstance(result, list)
        assert len(result) == 0
```

---

## Quick Reference Card

### Issue Severities
| Severity | Points | Action |
|----------|--------|--------|
| Critical | -20 | Stop and fix NOW |
| Warning | -7 | Fix soon |
| Info | -2 | Consider optimizing |

### Common Fixes

| Problem | Quick Fix |
|---------|-----------|
| Loss explosion | `clip_grad_norm_(model.parameters(), 1.0)` |
| Overfitting | `nn.Dropout(0.5)`, `weight_decay=0.01` |
| Memory leak | Use `loss.item()` not `loss` |
| Dead neurons | `nn.init.kaiming_normal_(layer.weight)` |
| No GPU | `model.to('cuda')`, `x.to('cuda')` |
| Vanishing gradients | Add skip connections, use LayerNorm |
| Wasted compute | Implement early stopping |

### Health Score Guide
- **90-100**: Excellent! Training is healthy
- **70-89**: Good, but has some issues to address
- **50-69**: Warning - several problems detected
- **0-49**: Critical - training likely broken

---

## Summary

The ML Diagnostics Engine:

1. **Collects** training metrics via the Observer
2. **Analyzes** 30+ potential issues automatically
3. **Reports** problems with severity, description, and suggestions
4. **Tracks** sustainability (GreenAI) including CO2 emissions
5. **Provides** architecture-specific checks for CNNs

**Key files:**
- `backend/diagnostics/engine.py` - All diagnostic checks
- `backend/diagnostics/schemas.py` - Output data models
- `backend/routers/diagnostics.py` - API endpoints
- `neural_network/observer.py` - Data collection

**Remember:** The goal is to help you train better models faster while using less energy! 🌱
