from transformers import HfArgumentParser
import torch

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import RayVllmGenerator
from utils.arguments import (
    ModelConfig,
    InfrastructureConfig,
    InferenceConfig,
    GenerationConfig,
    PromptConfig,
    BenchmarkConfig
)


def main():
    parser = HfArgumentParser([
        ModelConfig,
        InfrastructureConfig,
        InferenceConfig,
        GenerationConfig,
        PromptConfig,
        BenchmarkConfig
    ])
    model_config, infra_config, inference_config, generation_config, prompt_config, benchmark_config = \
        parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
    )
    # Benchmark.print_benchmark_table()

    # 2. Initialize Ray + vLLM generator (Multi-Node Support)
    generator = RayVllmGenerator(
        model_name_or_path=model_config.model_path,
        checkpoint_path=model_config.checkpoint_path,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        max_model_len=model_config.max_model_len,
        max_logprobs=model_config.max_logprobs,
        gpu_memory_utilization=infra_config.gpu_memory_utilization,
        tensor_parallel_size=infra_config.tensor_parallel_size,
        ray_address=infra_config.ray_address,  # Ray cluster address
        allow_cross_node_tensor_parallel=infra_config.allow_cross_node_tensor_parallel,  # Cross-node TP
        num_gpus=infra_config.num_gpus,
        gpu_ids=infra_config.gpu_ids,
        force_enable_optimizations=inference_config.force_enable_optimizations,
        force_disable_optimizations=inference_config.force_disable_optimizations,
        worker_batch_size=inference_config.worker_batch_size,
        task_types=benchmark_config.task_types
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        # Generation parameters
        enable_thinking=prompt_config.enable_thinking,
        num_beams=generation_config.num_beams,
        use_compact_beam_search=generation_config.use_compact_beam_search,
        num_return_sequences=generation_config.num_return_sequences,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        presence_penalty=generation_config.presence_penalty,
        num_return_thinking_sequences=generation_config.num_return_thinking_sequences,
        sample_size=benchmark_config.sample_size,
    )

    # 4. Release GPU memory occupied by vLLM
    console.print("\nReleasing vLLM GPU memory...", style=warning_style)
    generator.cleanup()
    del generator
    import gc
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    console.print("✓ GPU memory release completed\n", style=success_style)

    # 5. Calculate evaluation metrics
    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=benchmark_config.task_types
    )


if __name__ == "__main__":
    main()

