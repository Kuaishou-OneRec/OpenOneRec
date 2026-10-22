from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model loading and initialization parameters"""
    model_path: str = field(
        metadata={"help": "Model path or HuggingFace model name (e.g., Qwen/Qwen2-7B)", "required": True}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "PT checkpoint path (optional, for loading .pt format models, will auto-convert to HuggingFace format)"}
    )
    dtype: str = field(
        default='bfloat16',
        metadata={"help": "Model data type: auto, half, float16, bfloat16, float, float32"}
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum model length (optional, for limiting context length)"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code"}
    )
    max_logprobs: int = field(
        default=384,
        metadata={"help": "Maximum number of log probabilities to return (for beam search and logprob extraction)"}
    )


@dataclass
class InfrastructureConfig:
    """Hardware and distributed computing configuration"""
    # GPU allocation
    num_gpus: Optional[int] = field(
        default=None,
        metadata={"help": "Number of GPUs to use (default uses all visible GPUs)"}
    )
    gpu_ids: Optional[List[int]] = field(
        default=None,
        metadata={"help": "List of GPU IDs to use (e.g., [0,2,4], default uses all visible GPUs)"}
    )
    gpu_memory_utilization: float = field(
        default=0.5,
        metadata={"help": "GPU memory utilization (0-1, recommended 0.8)"}
    )
    # Parallelism
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size (default 1, single GPU per worker)"}
    )
    allow_cross_node_tensor_parallel: bool = field(
        default=False,
        metadata={"help": "Allow tensor parallelism across different nodes (not recommended due to network latency)"}
    )
    # Ray cluster
    ray_address: Optional[str] = field(
        default="auto",
        metadata={"help": "Ray cluster address: 'auto' (auto-detect), 'local' (single machine), or 'ray://head_ip:10001' (specific cluster address)"}
    )


@dataclass
class InferenceConfig:
    """Inference execution and optimization parameters"""
    # vLLM optimizations (chunked_prefill, prefix_caching)
    force_enable_optimizations: bool = field(
        default=False,
        metadata={"help": "Force enable chunked_prefill and prefix_caching for all tasks (overrides task-specific settings)"}
    )
    force_disable_optimizations: bool = field(
        default=False,
        metadata={"help": "Force disable chunked_prefill and prefix_caching for all tasks (overrides task-specific settings)"}
    )
    # Batch processing
    worker_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for each worker to process prompts (reduce this if KV cache is insufficient)"}
    )


@dataclass
class GenerationConfig:
    """Text generation parameters (sampling, beam search)"""
    use_compact_beam_search: bool = field(
        default=False,
        metadata={"help": "Use suffix-only beam state and lazy full-text decoding (vLLM 0.7.3, text inputs)"}
    )
    # Beam search
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "Number of beams for beam search"}
    )
    # Sampling
    num_return_sequences: Optional[int] = field(
        default=None,
        metadata={"help": "Number of sequences to return"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Sampling temperature"}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Top-p (nucleus) sampling probability"}
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Top-k sampling"}
    )
    presence_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "Presence penalty for sampling (-2.0 to 2.0, positive values penalize new tokens based on whether they appear in the text so far)"}
    )
    # Two-stage generation (thinking mode)
    num_return_thinking_sequences: Optional[int] = field(
        default=None,
        metadata={"help": "Number of thinking candidates to generate in stage 1"}
    )


@dataclass
class PromptConfig:
    """Prompt formatting and template parameters"""
    # Thinking mode (affects both template and generation)
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for apply_chat_template (overrides task config if set)"}
    )


@dataclass
class BenchmarkConfig:
    """Benchmark execution and evaluation parameters"""
    # Task selection
    task_types: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Task name list (e.g., item_understand rec_reason)"}
    )
    sample_size: Optional[str] = field(
        default=None,
        metadata={"help": "Sample size for evaluation (e.g., 'full' for all data, or a number like '100')"}
    )
    splits: List[str] = field(
        default_factory=lambda: ['test'],
        metadata={"help": "Dataset split list"}
    )
    # Data I/O
    data_dir: str = field(
        default='./data',
        metadata={"help": "Data directory path"}
    )
    output_dir: str = field(
        default='./results',
        metadata={"help": "Output directory for results"}
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing results"}
    )
