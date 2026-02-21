"""
Pydantic models for the Observer JSON report.

Generated from architecture.json (observer/v1 schema).
Use ObserverReport.model_validate(json_data) to parse and validate
an exported observer JSON file.

    import json
    from schema import ObserverReport

    with open("observer_reports/run.json") as f:
        report = ObserverReport.model_validate(json.load(f))

    # Typed access to everything
    print(report.session.device)
    print(report.model_architecture.layer_graph.nodes[0].weight_shape)
    for epoch in report.epochs:
        print(epoch.loss.train_mean, epoch.gradients.health)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# =====================================================================
# Enums
# =====================================================================

class LayerCategory(str, Enum):
    LINEAR = "linear"
    EMBEDDING = "embedding"
    NORMALIZATION = "normalization"
    ACTIVATION = "activation"
    REGULARIZATION = "regularization"
    CONVOLUTION = "convolution"
    ATTENTION = "attention"
    CONTAINER = "container"
    CUSTOM = "custom"


class EdgeRelation(str, Enum):
    CONTAINS = "contains"
    DATA_FLOW = "data_flow"


class GradientHealth(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"


class GradientIssueType(str, Enum):
    EXPLODING = "exploding"
    VANISHING = "vanishing"
    NAN = "NaN"
    INF = "Inf"


# =====================================================================
# Session
# =====================================================================

class SessionConfig(BaseModel):
    """Snapshot of ObserverConfig used for the run."""
    model_config = {"extra": "allow"}

    track_profiler: Optional[bool] = None
    track_gradients: Optional[bool] = None
    track_activations: Optional[bool] = None
    track_memory: Optional[bool] = None
    track_throughput: Optional[bool] = None
    track_loss: Optional[bool] = None
    track_weights: Optional[bool] = None
    track_layer_graph: Optional[bool] = None
    track_attention_entropy: Optional[bool] = None
    track_console_logs: Optional[bool] = None
    track_error_logs: Optional[bool] = None
    track_hyperparameters: Optional[bool] = None
    track_system_resources: Optional[bool] = None
    include_schema: Optional[bool] = None


class Session(BaseModel):
    """Run-level metadata captured at init and close."""
    api_key_prefix: str = Field(description="First 8 chars of the API key (masked).")
    project_id: str = Field(description="Project identifier.")
    run_id: str = Field(description="Unique run ID = project_id + run_name.")
    run_name: str = Field(description="Human-readable run label.")
    started_at: str = Field(description="ISO-8601 timestamp when Observer was created.")
    ended_at: Optional[str] = Field(None, description="ISO-8601 timestamp when close() was called.")
    device: str = Field(description="Compute device name (e.g. 'cpu', 'NVIDIA A100').")
    cuda_available: bool = Field(description="Whether CUDA was available.")
    pytorch_version: str = Field(description="PyTorch version string.")
    config: SessionConfig = Field(description="Snapshot of the ObserverConfig used for this run.")


# =====================================================================
# Model Architecture
# =====================================================================

class LayerInfo(BaseModel):
    """Basic layer info in the layers map."""
    type: str = Field(description="Module class name.")
    parameters: int = Field(description="Number of own parameters.")
    pct_of_total: float = Field(description="Percentage of total model parameters.")


class ModuleTree(BaseModel):
    """Recursive tree mirroring nn.Module nesting."""
    type: str = Field(description="Module class name.")
    children: Optional[Dict[str, ModuleTree]] = Field(None, description="Child modules.")


# -- Layer Graph (track_layer_graph=True) --

class ParameterDetail(BaseModel):
    """Per-tensor parameter details."""
    shape: List[int] = Field(description="Tensor dimensions.")
    numel: int = Field(description="Total number of elements.")
    dtype: str = Field(description="Data type (e.g. 'torch.float32').")
    requires_grad: bool = Field(description="Whether this tensor is trainable.")
    mean: float = Field(description="Mean of initial weight values.")
    std: float = Field(description="Std dev of initial weight values.")
    norm: float = Field(description="L2 norm of initial weight values.")


class BufferDetail(BaseModel):
    """Non-parameter persistent tensor (e.g. causal mask, running stats)."""
    shape: List[int] = Field(description="Buffer tensor shape.")
    dtype: str = Field(description="Buffer data type.")


class GraphNode(BaseModel):
    """A single module/layer in the architecture graph."""
    model_config = {"extra": "allow"}

    id: str = Field(description="Fully-qualified module name (e.g. 'blocks.0.sa.heads.0.key').")
    type: str = Field(description="Class name (e.g. 'Linear', 'LayerNorm', 'Head').")
    is_container: bool = Field(description="True if the module has child modules.")
    category: LayerCategory = Field(description="Semantic category of this layer.")

    # -- Neuron / dimension info (conditional by layer type) --
    neurons: Optional[Union[int, List[int]]] = Field(None, description="Output neuron count / dimension.")

    # Linear
    in_features: Optional[int] = Field(None, description="Input dimension (Linear).")
    out_features: Optional[int] = Field(None, description="Output dimension (Linear).")
    has_bias: Optional[bool] = Field(None, description="Whether the layer has a bias vector.")
    weight_shape: Optional[List[int]] = Field(None, description="Shape of the weight tensor, e.g. [512, 128].")
    bias_shape: Optional[List[int]] = Field(None, description="Shape of the bias tensor, e.g. [512].")

    # Embedding
    num_embeddings: Optional[int] = Field(None, description="Vocabulary size (Embedding).")
    embedding_dim: Optional[int] = Field(None, description="Embedding vector dimension (Embedding).")

    # Normalization
    normalized_shape: Optional[List[int]] = Field(None, description="Shape being normalized (LayerNorm).")
    eps: Optional[float] = Field(None, description="Epsilon for numerical stability (LayerNorm/BatchNorm).")
    has_weight: Optional[bool] = Field(None, description="Whether LayerNorm/BatchNorm has learnable weight.")
    num_features: Optional[int] = Field(None, description="Number of features (BatchNorm).")
    momentum: Optional[float] = Field(None, description="Momentum for running stats (BatchNorm).")
    affine: Optional[bool] = Field(None, description="Whether BatchNorm has learnable parameters.")

    # Dropout
    p: Optional[float] = Field(None, description="Dropout probability.")

    # Activation
    inplace: Optional[bool] = Field(None, description="Whether activation is inplace.")

    # Convolution
    in_channels: Optional[int] = Field(None, description="Input channels (Conv).")
    out_channels: Optional[int] = Field(None, description="Output channels (Conv).")
    kernel_size: Optional[List[int]] = Field(None, description="Kernel dimensions (Conv).")
    stride: Optional[List[int]] = Field(None, description="Stride dimensions (Conv).")
    padding: Optional[List[int]] = Field(None, description="Padding dimensions (Conv).")

    # MultiheadAttention
    embed_dim: Optional[int] = Field(None, description="Total embedding dimension (MultiheadAttention).")
    num_heads: Optional[int] = Field(None, description="Number of attention heads (MultiheadAttention).")
    head_dim: Optional[int] = Field(None, description="Per-head dimension (MultiheadAttention).")

    # Container
    num_children: Optional[int] = Field(None, description="Number of child modules (Sequential/ModuleList).")
    padding_idx: Optional[int] = Field(None, description="Padding index (Embedding).")

    # Weight & buffer details
    parameters: Optional[Dict[str, ParameterDetail]] = Field(None, description="Per-tensor parameter details.")
    total_params: Optional[int] = Field(None, description="Sum of numel across all own parameters.")
    buffers: Optional[Dict[str, BufferDetail]] = Field(None, description="Non-parameter persistent tensors.")


class GraphEdge(BaseModel):
    """Connection between two modules."""
    source: str = Field(description="Source module ID.")
    target: str = Field(description="Target module ID.")
    relation: EdgeRelation = Field(description="'contains' or 'data_flow'.")


class DimensionFlowEntry(BaseModel):
    """How dimensions transform at a single layer."""
    model_config = {"extra": "allow"}

    layer: str = Field(description="Module name.")
    type: str = Field(description="Layer class (Linear, Embedding, LayerNorm, Conv*).")

    # Linear
    in_: Optional[int] = Field(None, alias="in", description="Input dimension (Linear).")
    out: Optional[int] = Field(None, description="Output dimension (Linear).")

    # Embedding
    vocab: Optional[int] = Field(None, description="Vocabulary size (Embedding).")
    dim: Optional[int] = Field(None, description="Embedding dimension (Embedding).")

    # LayerNorm
    shape: Optional[List[int]] = Field(None, description="Normalized shape (LayerNorm).")

    # Conv
    in_ch: Optional[int] = Field(None, description="Input channels (Conv).")
    out_ch: Optional[int] = Field(None, description="Output channels (Conv).")
    kernel: Optional[List[int]] = Field(None, description="Kernel size (Conv).")


class LayerGraph(BaseModel):
    """Full layer graph for architecture visualization."""
    nodes: List[GraphNode] = Field(description="Ordered list of every module in the model.")
    edges: List[GraphEdge] = Field(description="Connections between modules.")
    sequential_path: List[str] = Field(description="Ordered compute-layer IDs in forward-pass order.")
    dimension_flow: List[DimensionFlowEntry] = Field(description="How tensor dimensions transform through the network.")
    total_compute_layers: int = Field(description="Count of non-container leaf compute layers.")
    total_nodes: int = Field(description="Total node count in the graph.")
    total_edges: int = Field(description="Total edge count in the graph.")


class ModelArchitecture(BaseModel):
    """Static model analysis captured once at register_model()."""
    total_parameters: int = Field(description="Total number of parameters.")
    trainable_parameters: int = Field(description="Parameters with requires_grad=True.")
    frozen_parameters: int = Field(description="Parameters with requires_grad=False.")
    num_parameter_layers: int = Field(description="Count of modules that own at least one parameter.")
    layers: Dict[str, LayerInfo] = Field(description="Map of module_name -> basic layer info.")
    module_tree: ModuleTree = Field(description="Recursive tree mirroring nn.Module nesting.")
    layer_graph: Optional[LayerGraph] = Field(None, description="Full layer graph. Present when track_layer_graph=True.")


# =====================================================================
# Epoch — Loss
# =====================================================================

class EpochLoss(BaseModel):
    """Loss statistics for a single epoch."""
    train_mean: float = Field(description="Mean batch loss across all batches.")
    train_min: float = Field(description="Minimum batch loss.")
    train_max: float = Field(description="Maximum batch loss.")
    train_first: float = Field(description="Loss of the first batch.")
    train_last: float = Field(description="Loss of the last batch.")
    train_std: float = Field(description="Std dev of batch losses.")
    num_batches: int = Field(description="Number of batches in this epoch.")
    val: Optional[Dict[str, float]] = Field(None, description="Validation metrics { metric_name: value }.")


# =====================================================================
# Epoch — Throughput
# =====================================================================

class EpochThroughput(BaseModel):
    """Speed metrics for a single epoch."""
    samples_processed: int = Field(description="Total samples processed this epoch.")
    tokens_processed: int = Field(description="Total tokens processed this epoch.")
    samples_per_second: float = Field(description="Samples / wall-clock second.")
    tokens_per_second: float = Field(description="Tokens / wall-clock second.")
    batches_per_second: float = Field(description="Batches / wall-clock second.")
    seconds_per_batch: float = Field(description="Average wall-clock seconds per batch.")


# =====================================================================
# Epoch — Gradients
# =====================================================================

class GradientLayerStats(BaseModel):
    """Per-parameter gradient statistics."""
    norm: float = Field(description="L2 norm of the gradient.")
    mean: float = Field(description="Mean gradient value.")
    std: float = Field(description="Std dev of gradient values.")
    abs_mean: float = Field(description="Mean of absolute gradient values.")
    min: float = Field(description="Minimum gradient value.")
    max: float = Field(description="Maximum gradient value.")


class GradientIssue(BaseModel):
    """A detected gradient health issue."""
    layer: str = Field(description="Parameter name.")
    issue: GradientIssueType = Field(description="Issue type.")
    norm: Optional[float] = Field(None, description="Gradient norm (for exploding/vanishing).")


class EpochGradients(BaseModel):
    """Gradient health snapshot for a single epoch."""
    total_norm: float = Field(description="Global L2 norm across all parameter gradients.")
    num_layers: int = Field(description="Number of layers with gradients.")
    health: GradientHealth = Field(description="'healthy' or 'warning'.")
    issues: List[GradientIssue] = Field(default_factory=list, description="Detected gradient issues.")
    per_layer: Dict[str, GradientLayerStats] = Field(description="Per-parameter gradient stats.")


# =====================================================================
# Epoch — Weights
# =====================================================================

class WeightLayerStats(BaseModel):
    """Per-parameter weight distribution snapshot."""
    mean: float = Field(description="Mean weight value.")
    std: float = Field(description="Std dev of weight values.")
    norm: float = Field(description="L2 norm of weight values.")
    min: float = Field(description="Minimum weight value.")
    max: float = Field(description="Maximum weight value.")
    numel: int = Field(description="Number of elements.")


# =====================================================================
# Epoch — Activations
# =====================================================================

class ActivationStats(BaseModel):
    """Activation statistics for a single hooked layer."""
    mean: float = Field(description="Mean activation value.")
    std: float = Field(description="Std dev of activation values.")
    min: float = Field(description="Minimum activation value.")
    max: float = Field(description="Maximum activation value.")
    abs_mean: float = Field(description="Mean of absolute activation values.")
    dead_fraction: float = Field(description="Fraction of zero activations (dead neurons).")
    saturated_fraction: float = Field(description="Fraction of saturated activations.")
    shape: List[int] = Field(description="Output tensor shape.")


# =====================================================================
# Epoch — Attention Entropy
# =====================================================================

class AttentionEntropyStats(BaseModel):
    """Attention output entropy for a single head/layer."""
    output_entropy: float = Field(description="Entropy of the attention output distribution.")
    max_possible_entropy: float = Field(description="Maximum possible entropy (log of dim).")
    entropy_ratio: float = Field(description="output_entropy / max_possible_entropy.")
    output_sparsity: float = Field(description="Fraction of near-zero values in output.")


# =====================================================================
# Epoch — Profiler
# =====================================================================

class ProfilerOperation(BaseModel):
    """A single profiled operation."""
    model_config = {"extra": "allow"}

    name: str = Field(description="Operation name.")
    calls: int = Field(description="Number of calls.")
    cpu_time_us: int = Field(description="Total CPU time in microseconds.")
    cpu_time_ms: float = Field(description="Total CPU time in milliseconds.")
    avg_cpu_us: float = Field(description="Average CPU time per call in microseconds.")
    cuda_time_us: Optional[int] = Field(None, description="Total CUDA time in microseconds.")
    cuda_time_ms: Optional[float] = Field(None, description="Total CUDA time in milliseconds.")
    cpu_mem_bytes: Optional[int] = Field(None, description="CPU memory usage in bytes.")
    cuda_mem_bytes: Optional[int] = Field(None, description="CUDA memory usage in bytes.")
    input_shapes: Optional[List[str]] = Field(None, description="Input tensor shapes.")


class ProfilerCategory(BaseModel):
    """Aggregated timing for an operation category."""
    cpu_time_ms: float = Field(description="Total CPU time in milliseconds.")
    cuda_time_ms: float = Field(description="Total CUDA time in milliseconds.")
    calls: int = Field(description="Total number of calls.")
    pct_cpu: float = Field(description="Percentage of total CPU time.")


class EpochProfiler(BaseModel):
    """PyTorch Profiler results from a single profiled step."""
    total_cpu_time_ms: float = Field(description="Total CPU time across all ops (ms).")
    total_cuda_time_ms: float = Field(description="Total CUDA time across all ops (ms).")
    forward_time_ms: float = Field(description="CPU time in forward pass (ms).")
    backward_time_ms: float = Field(description="CPU time in backward pass (ms).")
    fwd_bwd_ratio: float = Field(description="Forward time / backward time.")
    num_unique_ops: int = Field(description="Number of unique operator types.")
    top_operations: List[ProfilerOperation] = Field(description="Top N ops sorted by CPU time.")
    operation_categories: Dict[str, ProfilerCategory] = Field(description="Timing aggregated by category.")


# =====================================================================
# Epoch — Memory
# =====================================================================

class EpochMemory(BaseModel):
    """Memory usage snapshot."""
    model_config = {"extra": "allow"}

    process_rss_mb: Optional[float] = Field(None, description="Process RSS in MB (requires psutil).")
    process_vms_mb: Optional[float] = Field(None, description="Process VMS in MB (requires psutil).")
    cuda_allocated_mb: Optional[float] = Field(None, description="CUDA memory currently allocated (MB).")
    cuda_reserved_mb: Optional[float] = Field(None, description="CUDA memory currently reserved (MB).")
    cuda_peak_allocated_mb: Optional[float] = Field(None, description="Peak CUDA memory allocated this epoch (MB).")
    cuda_peak_reserved_mb: Optional[float] = Field(None, description="Peak CUDA memory reserved this epoch (MB).")


# =====================================================================
# Epoch — System
# =====================================================================

class EpochSystem(BaseModel):
    """System resource snapshot."""
    model_config = {"extra": "allow"}

    cpu_percent: Optional[float] = Field(None, description="CPU utilization %.")
    ram_total_gb: Optional[float] = Field(None, description="Total system RAM in GB.")
    ram_used_gb: Optional[float] = Field(None, description="Used system RAM in GB.")
    ram_percent: Optional[float] = Field(None, description="RAM utilization %.")
    gpu_name: Optional[str] = Field(None, description="GPU device name.")
    gpu_total_mem_mb: Optional[float] = Field(None, description="Total GPU memory (MB).")


# =====================================================================
# Epoch — Log Counts
# =====================================================================

class LogCounts(BaseModel):
    """Running count of captured log entries."""
    console: int = Field(description="Total console log entries so far.")
    error: int = Field(description="Total error/warning log entries so far.")


# =====================================================================
# Epoch (top-level)
# =====================================================================

class EpochRecord(BaseModel):
    """Complete telemetry for a single epoch."""
    epoch: int = Field(description="Epoch index.")
    timestamp: str = Field(description="ISO-8601 time when end_epoch() was called.")
    duration_seconds: float = Field(description="Wall-clock seconds for this epoch.")
    loss: Optional[EpochLoss] = Field(None, description="Loss statistics. Present when track_loss=True.")
    throughput: Optional[EpochThroughput] = Field(None, description="Speed metrics. Present when track_throughput=True.")
    gradients: Optional[EpochGradients] = Field(None, description="Gradient health. Present when track_gradients=True.")
    weights: Optional[Dict[str, WeightLayerStats]] = Field(None, description="Weight snapshots. Present when track_weights=True.")
    activations: Optional[Dict[str, ActivationStats]] = Field(None, description="Activation stats. Present when track_activations=True.")
    attention_entropy: Optional[Dict[str, AttentionEntropyStats]] = Field(None, description="Attention entropy. Present when track_attention_entropy=True.")
    profiler: Optional[EpochProfiler] = Field(None, description="Profiler results. Present when profile_step() was called.")
    memory: Optional[EpochMemory] = Field(None, description="Memory usage. Present when track_memory=True.")
    system: Optional[EpochSystem] = Field(None, description="System resources. Present when track_system_resources=True.")
    log_counts: LogCounts = Field(description="Running count of captured log entries.")


# =====================================================================
# Log entries
# =====================================================================

class LogEntry(BaseModel):
    """A captured log record."""
    ts: str = Field(description="ISO-8601 timestamp.")
    level: str = Field(description="Log level (INFO, WARNING, ERROR, etc.).")
    msg: str = Field(description="Log message.")
    module: str = Field(default="", description="Source module name.")
    lineno: int = Field(default=0, description="Source line number.")


# =====================================================================
# Summary
# =====================================================================

class LossTrend(BaseModel):
    """Loss progression across epochs."""
    first: float = Field(description="Train loss mean of first epoch.")
    last: float = Field(description="Train loss mean of last epoch.")
    best: float = Field(description="Lowest train loss mean across epochs.")
    worst: float = Field(description="Highest train loss mean across epochs.")
    delta: float = Field(description="first - last (positive = improvement).")
    improved: bool = Field(description="Whether last < first.")


class GradientHealthSummary(BaseModel):
    """Gradient health aggregated across epochs."""
    epochs_with_issues: int = Field(description="Epochs that had at least one gradient issue.")
    total_issues: int = Field(description="Cumulative gradient issues across all epochs.")


class ProfilerHighlight(BaseModel):
    """Key stats from the last profiled epoch."""
    fwd_bwd_ratio: Optional[float] = Field(None, description="Forward / backward time ratio.")
    top_op: Optional[str] = Field(None, description="Name of the most expensive operation.")
    top_op_pct: Optional[float] = Field(None, description="% of total CPU time for the top op.")


class Summary(BaseModel):
    """Aggregated summary computed at export time."""
    model_config = {"extra": "allow"}

    total_epochs: Optional[int] = Field(None, description="Number of epochs recorded.")
    total_duration_s: Optional[float] = Field(None, description="Sum of all epoch durations (seconds).")
    loss_trend: Optional[LossTrend] = Field(None, description="Loss progression across epochs.")
    gradient_health: Optional[GradientHealthSummary] = Field(None, description="Gradient health across epochs.")
    avg_tokens_per_sec: Optional[float] = Field(None, description="Average throughput across epochs.")
    profiler_highlight: Optional[ProfilerHighlight] = Field(None, description="Key stats from last profiled epoch.")
    status: Optional[str] = Field(None, description="'no_data' if no epochs were recorded.")


# =====================================================================
# Root — ObserverReport
# =====================================================================

class ObserverReport(BaseModel):
    """
    Top-level Pydantic model for the Observer JSON report.

    Usage:
        import json
        from schema import ObserverReport

        with open("observer_reports/run.json") as f:
            report = ObserverReport.model_validate(json.load(f))

        report.session.device
        report.model_architecture.layer_graph.nodes
        report.epochs[0].profiler.top_operations
    """
    model_config = {"extra": "allow"}

    schema_: Optional[Dict[str, Any]] = Field(None, alias="schema", description="Self-describing schema document.")
    session: Session = Field(description="Run-level metadata.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Training hyperparameters.")
    model_architecture: ModelArchitecture = Field(description="Static model analysis.")
    epochs: List[EpochRecord] = Field(default_factory=list, description="Per-epoch telemetry records.")
    console_logs: List[LogEntry] = Field(default_factory=list, description="Captured INFO+ log records.")
    error_logs: List[LogEntry] = Field(default_factory=list, description="Captured WARNING+ log records.")
    summary: Summary = Field(description="Aggregated summary.")
