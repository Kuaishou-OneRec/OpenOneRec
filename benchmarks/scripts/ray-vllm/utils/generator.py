import os
import ray
import math
import json
from typing import Dict, List, Any, Optional
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from benchmark.compact_beam_search import compact_beam_search

from benchmark.base_generator import Generator, RayMixin, VllmMixin, DISABLE_OPTIMIZATIONS_FOR_TASKS
from benchmark.checkpoint_utils import export_pt_to_safetensor
from benchmark.console import *


class VllmWorker:
    """
    vLLM Worker that can use one or more GPUs
    
    Each Worker is responsible for:
    - Loading one vLLM model instance (potentially across multiple GPUs with tensor parallelism)
    - Processing inference tasks assigned to it
    - Returning generation results
    """
    
    def __init__(
        self,
        worker_id: int,
        model_path: str,
        gpu_ids: List[int],
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        enable_optimizations: bool = True,
        **kwargs
    ):
        """
        Args:
            worker_id: Worker ID
            model_path: Model path (converted HuggingFace format)
            gpu_ids: List of GPU IDs assigned to this worker
            gpu_memory_utilization: GPU memory utilization
            trust_remote_code: Whether to trust remote code
            dtype: Data type
            max_model_len: Maximum model length
            tensor_parallel_size: Tensor parallel size (must match len(gpu_ids))
            enable_optimizations: Whether to enable chunked_prefill and prefix_caching
            **kwargs: Other vLLM parameters
        """
        self.worker_id = worker_id
        self.gpu_ids = gpu_ids
        
        # Set environment variable so current process only sees specified GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        opt_status = "optimized" if enable_optimizations else "standard"
        gpu_str = ",".join(map(str, gpu_ids))
        print(f"  [Worker {worker_id}] Initializing ({opt_status})... (GPU {gpu_str}, TP={tensor_parallel_size})")
        
        # Initialize vLLM
        vllm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "enable_chunked_prefill": enable_optimizations,
            "enable_prefix_caching": enable_optimizations,
            "max_logprobs": kwargs.get("max_logprobs", 384),  # Support beam search, need large enough logprobs
        }

        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len

        vllm_kwargs.update(kwargs)
        
        try:
            self.llm = LLM(**vllm_kwargs)
            self.tokenizer = self.llm.get_tokenizer()
            print(f"  [Worker {worker_id}] ✓ Initialized successfully (GPU {gpu_str}, TP={tensor_parallel_size})")
        except Exception as e:
            print(f"  [Worker {worker_id}] ✗ Initialization failed: {e}")
            raise
    
    def get_model_parameters(self) -> Optional[float]:
        """
        Get model parameter count from the worker's vLLM instance
        
        Returns:
            float: Total number of parameters, or None if unable to count
        """
        try:
            model_executor = self.llm.llm_engine.model_executor
            if hasattr(model_executor, 'driver_worker'):
                model = model_executor.driver_worker.model_runner.model
            else:
                model = model_executor.model
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            return float(total_params)
        except Exception as e:
            print(f"  [Worker {self.worker_id}] Warning: Failed to count parameters: {e}")
            return None
    
    def generate_batch(
        self,
        prompts: Dict[str, str],
        sampling_params: Dict[str, Any],
        worker_batch_size: int = 8
    ) -> tuple:
        """
        Batch text generation (internal batch processing to avoid vLLM scheduler issues)

        Args:
            prompts: {sample_id: prompt_text}
            sampling_params: Sampling parameter dictionary
            worker_batch_size: Worker internal batch size (default 8)

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [cum_logprob_1, cum_logprob_2, ...]} (only for beam search)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        import time
        stage_start_time = time.time()

        if not prompts:
            return ({}, {}, {})

        if sampling_params.get("use_compact_beam_search") and not sampling_params.get("use_beam_search"):
            raise ValueError("use_compact_beam_search requires beam search")

        # Determine whether to use BeamSearchParams or SamplingParams based on parameters
        if sampling_params.get("use_beam_search", False):
            # Beam search mode
            params_dict = {
                "beam_width": sampling_params.get("beam_width", 1),
                "max_tokens": sampling_params.get("max_tokens", 128),
            }
            sp = BeamSearchParams(**params_dict)
        else:
            # Sampling mode - remove parameters not belonging to SamplingParams
            # stop parameter is already included in the dict comprehension
            params_dict = {k: v for k, v in sampling_params.items()
                          if k not in ["use_beam_search", "beam_width", "return_logprobs", "use_compact_beam_search"]}

            # If return_logprobs is enabled, add logprobs parameter
            if sampling_params.get("return_logprobs", False):
                params_dict["logprobs"] = 1  # Enable logprobs for cumulative calculation

            sp = SamplingParams(**params_dict)

        # Prepare input
        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())

        # Batch processing to avoid vLLM scheduler issues
        all_results = {}
        all_logprobs = {}  # Store cum_logprobs for beam search
        all_mfu_stats = {}  # Store MFU statistics for MFU calculation
        num_batches = (len(sample_ids) + worker_batch_size - 1) // worker_batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * worker_batch_size
            end_idx = min(start_idx + worker_batch_size, len(sample_ids))
            
            batch_sample_ids = sample_ids[start_idx:end_idx]
            batch_prompt_texts = prompt_texts[start_idx:end_idx]
            
            try:
                # If using beam search, need to record each prompt's length
                batch_prompt_lengths = []
                if isinstance(sp, BeamSearchParams):
                    for text in batch_prompt_texts:
                        prompt_tokens = self.tokenizer.encode(text, add_special_tokens=True)
                        batch_prompt_lengths.append(len(prompt_tokens))
                
                # Choose different generation method based on parameter type
                if isinstance(sp, BeamSearchParams):
                    # Beam search needs to use beam_search method
                    # beam_search input format is [{"prompt": "text"}]
                    batch_prompt_dicts = [{"prompt": text} for text in batch_prompt_texts]
                    if sampling_params.get("use_compact_beam_search"):
                        batch_outputs = compact_beam_search(self.llm, batch_prompt_dicts, sp)
                    else:
                        batch_outputs = self.llm.beam_search(batch_prompt_dicts, sp)
                else:
                    # Sampling mode uses generate method
                    batch_outputs = self.llm.generate(batch_prompt_texts, sp)
                
                # Organize results
                for idx, (sample_id, output) in enumerate(zip(batch_sample_ids, batch_outputs)):
                    if isinstance(sp, BeamSearchParams):
                        # Beam search returns complete token IDs (including prompt), need to remove prompt part before decoding
                        prompt_length = batch_prompt_lengths[idx]
                        generated_texts = [
                            self.tokenizer.decode(seq.tokens[prompt_length:], skip_special_tokens=True)
                            for seq in output.sequences
                        ]
                        # Extract cum_logprob for each sequence
                        cum_logprobs = [seq.cum_logprob for seq in output.sequences]
                        all_results[sample_id] = generated_texts
                        all_logprobs[sample_id] = cum_logprobs

                        # Collect MFU stats for beam search
                        input_tokens = prompt_length
                        output_tokens_list = [len(seq.tokens) - prompt_length for seq in output.sequences]
                        all_mfu_stats[sample_id] = {
                            "input_tokens": [input_tokens],
                            "output_tokens": [sum(output_tokens_list)]
                        }
                    else:
                        generated_texts = [out.text for out in output.outputs]
                        all_results[sample_id] = generated_texts

                        # If return_logprobs is enabled, calculate cumulative logprobs
                        if sampling_params.get("return_logprobs", False):
                            cum_logprobs = []
                            for out in output.outputs:
                                # Calculate cumulative logprob by summing all token logprobs
                                cum_logprob = 0.0
                                if out.logprobs and out.token_ids:
                                    # Iterate through each position and get the logprob of the actual generated token
                                    for i, token_logprobs in enumerate(out.logprobs):
                                        if token_logprobs and i < len(out.token_ids):
                                            # Get the actual token ID that was generated at this position
                                            actual_token_id = out.token_ids[i]
                                            # Look up the logprob for this specific token
                                            if actual_token_id in token_logprobs:
                                                cum_logprob += token_logprobs[actual_token_id].logprob
                                cum_logprobs.append(cum_logprob)
                            all_logprobs[sample_id] = cum_logprobs

                        # Collect MFU stats for sampling mode
                        prompt_text = batch_prompt_texts[idx]
                        input_tokens = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
                        output_tokens_list = [len(out.token_ids) for out in output.outputs]
                        all_mfu_stats[sample_id] = {
                            "input_tokens": [input_tokens],
                            "output_tokens": [sum(output_tokens_list)]
                        }
                
            except Exception as e:
                if sampling_params.get("use_compact_beam_search"):
                    raise RuntimeError("Compact beam search failed; refusing empty-output fallback") from e
                # When a single batch fails, return empty string and print detailed error
                import traceback
                print(f"\n[Worker {self.worker_id}] Batch {batch_idx}/{num_batches} generation failed:")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error message: {str(e)}")
                print(f"  Batch size: {len(batch_sample_ids)}")
                if batch_prompt_texts:
                    prompt_lens = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in batch_prompt_texts]
                    print(f"  Prompt token length range: min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens)/len(prompt_lens):.1f}")
                print(f"  Full stack trace:\n{traceback.format_exc()}")
                
                num_return = sampling_params.get("n", 1)
                if sampling_params.get("use_beam_search", False):
                    num_return = sampling_params.get("beam_width", 1)
                for sample_id in batch_sample_ids:
                    all_results[sample_id] = [""] * num_return
                    # If beam search, also set empty logprobs
                    if sampling_params.get("use_beam_search", False):
                        all_logprobs[sample_id] = [0.0] * num_return
                    # Don't include failed samples in MFU stats (they would have times=[0.0] which breaks MFU calculation)

        # Calculate stage time
        stage_elapsed_time = time.time() - stage_start_time

        # Add time to all samples (same time for all samples in this worker)
        for sample_id in all_mfu_stats:
            all_mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        return (all_results, all_logprobs, all_mfu_stats)
    
    def extract_token_logprobs_batch(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        sampling_params: Dict[str, Any],
        worker_batch_size: int = 8
    ) -> tuple:
        """
        Extract logprobs for specific target tokens

        Args:
            prompts: {sample_id: prompt_text}
            target_tokens: List of target tokens (e.g., ["是", "否"])
            sampling_params: Sampling parameter dictionary
            worker_batch_size: Worker internal batch size

        Returns:
            Tuple of two dicts:
            - First dict: {sample_id: [json_string]} where json_string is formatted probabilities
            - Second dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}}
        """
        import time
        stage_start_time = time.time()

        if not prompts:
            return ({}, {})
        
        # Get token IDs for target tokens
        target_token_ids = {}
        for token in target_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                target_token_ids[token] = token_ids[0]
            else:
                print(f"  [Worker {self.worker_id}] Warning: Token '{token}' is encoded as multiple tokens: {token_ids}")
                # For multi-token case, we only use the first token for now
                target_token_ids[token] = token_ids[0]
        
        # Build sampling parameters with logprobs enabled
        params_dict = {
            "n": sampling_params.get("n", 1),
            "max_tokens": sampling_params.get("max_tokens", 1),
            "temperature": sampling_params.get("temperature", 1.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "top_k": sampling_params.get("top_k", -1),
            "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
            "presence_penalty": sampling_params.get("presence_penalty", 0.0),
            "frequency_penalty": sampling_params.get("frequency_penalty", 0.0),
            "logprobs": sampling_params.get("logprobs", 10),
        }
        sp = SamplingParams(**params_dict)
        
        # Prepare input
        sample_ids = list(prompts.keys())
        prompt_texts = list(prompts.values())
        
        # Batch processing
        all_results = {}
        all_mfu_stats = {}
        num_batches = (len(sample_ids) + worker_batch_size - 1) // worker_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * worker_batch_size
            end_idx = min(start_idx + worker_batch_size, len(sample_ids))

            batch_sample_ids = sample_ids[start_idx:end_idx]
            batch_prompt_texts = prompt_texts[start_idx:end_idx]

            try:
                # Generate with logprobs
                batch_outputs = self.llm.generate(batch_prompt_texts, sp)

                # Extract logprobs for target tokens
                for idx, (sample_id, output) in enumerate(zip(batch_sample_ids, batch_outputs)):
                    token_probs = {}

                    # Get logprobs from the first generated token
                    if output.outputs and len(output.outputs) > 0:
                        first_output = output.outputs[0]
                        if first_output.logprobs and len(first_output.logprobs) > 0:
                            # Get logprobs dict for the first token
                            first_token_logprobs = first_output.logprobs[0]

                            # Extract probabilities for target tokens
                            for token, token_id in target_token_ids.items():
                                if token_id in first_token_logprobs:
                                    logprob = first_token_logprobs[token_id].logprob
                                    prob = math.exp(logprob)
                                    token_probs[token] = prob
                                else:
                                    # Token not in top-k, assign very small probability
                                    token_probs[token] = 1e-10

                    all_results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]

                    prompt_text = batch_prompt_texts[idx]
                    input_tokens = len(self.tokenizer.encode(prompt_text, add_special_tokens=True))
                    # Classification only generates 1 token
                    output_tokens = 1
                    all_mfu_stats[sample_id] = {
                        "input_tokens": [input_tokens],
                        "output_tokens": [output_tokens]
                    }

            except Exception as e:
                import traceback
                print(f"\n[Worker {self.worker_id}] Batch {batch_idx}/{num_batches} logprobs extraction failed:")
                print(f"  Error: {str(e)}")
                print(f"  Full stack trace:\n{traceback.format_exc()}")

                for sample_id in batch_sample_ids:
                    token_probs = {token: 0.0 for token in target_tokens}
                    all_results[sample_id] = [json.dumps(token_probs, ensure_ascii=False)]
                    # Don't include failed samples in MFU stats

        stage_elapsed_time = time.time() - stage_start_time
        for sample_id in all_mfu_stats:
            all_mfu_stats[sample_id]["times"] = [stage_elapsed_time]

        return (all_results, all_mfu_stats)


class RayVllmGenerator(RayMixin, VllmMixin, Generator):
    """
    Ray-based Multi-GPU vLLM Generator (Data Parallel)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: Optional[str] = None,
        num_return_sequences: int = 2,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        max_model_len: Optional[int] = None,
        max_logprobs: int = 384,
        tensor_parallel_size: int = 1,
        num_gpus: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
        task_types: Optional[List[str]] = None,
        force_enable_optimizations: bool = False,
        force_disable_optimizations: bool = False,
        worker_batch_size: int = 4,
        ray_address: Optional[str] = "auto",
        allow_cross_node_tensor_parallel: bool = False,
        **kwargs
    ):
        """
        Args:
            model_name_or_path: Model name or path
            checkpoint_path: PT checkpoint path (optional)
            num_return_sequences: Number of candidate sequences per prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            presence_penalty: Presence penalty (penalizes tokens that appeared in the text)
            frequency_penalty: Frequency penalty (penalizes tokens based on frequency)
            do_sample: Whether to sample
            gpu_memory_utilization: GPU memory utilization
            trust_remote_code: Whether to trust remote code
            dtype: Model data type
            max_model_len: Maximum model length
            max_logprobs: Maximum number of log probabilities to return (for beam search and logprob extraction)
            tensor_parallel_size: Tensor parallel size (default 1, single GPU per worker)
            num_gpus: Number of GPUs to use (default uses all cluster GPUs)
            gpu_ids: List of GPU IDs to use (only for single-node mode)
            task_types: List of task types to evaluate (for auto optimization control)
            force_enable_optimizations: Force enable optimizations for all tasks
            force_disable_optimizations: Force disable optimizations for all tasks
            worker_batch_size: Batch size for each worker (reduce if KV cache is insufficient)
            ray_address: Ray cluster address ('auto', 'local', or specific address)
            allow_cross_node_tensor_parallel: Allow tensor parallel across nodes (not recommended)
            **kwargs: Other parameters
        """
        super().__init__(
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            do_sample=do_sample,
            **kwargs
        )
        
        self.model_name = model_name_or_path
        self.checkpoint_path = checkpoint_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_batch_size = worker_batch_size
        self.task_types = task_types or []
        self.force_enable_optimizations = force_enable_optimizations
        self.force_disable_optimizations = force_disable_optimizations
        self.ray_address = ray_address
        self.allow_cross_node_tensor_parallel = allow_cross_node_tensor_parallel
        self.num_gpus = num_gpus
        self.gpu_ids = gpu_ids

        console.print(
            "\nLoading Model\n",
            style=head_style_2,
            justify="center",
        )
        console.print(
            f"  Using Ray + vLLM (Multi-Node) to load model: [cyan]{model_name_or_path}[/cyan]",
            style=subhead_style_2,
        )
        
        # 1. Initialize Ray cluster connection
        self._initialize_ray_cluster()
        
        # 2. Determine GPUs to use (from cluster)
        all_gpu_ids = self._determine_gpu_ids_from_cluster()
        
        # 3. Group GPUs for workers (ensuring same-node constraint if needed)
        self.worker_gpu_groups, self.worker_node_assignments = self._group_gpus_for_workers(
            all_gpu_ids, tensor_parallel_size
        )
        num_workers = len(self.worker_gpu_groups)
        
        # Display cluster and GPU information
        self._display_cluster_info(all_gpu_ids, num_workers)
        
        # 4. Handle PT checkpoint (main process converts)
        if checkpoint_path:
            console.print(
                f"  checkpoint: [yellow]{checkpoint_path}[/yellow]",
                style=subhead_style_2,
            )
            console.print(
                "  [yellow]Converting PT checkpoint to HuggingFace format in main process...[/yellow]",
                style=subhead_style_2,
            )
            model_path = export_pt_to_safetensor(
                config_path=model_name_or_path,
                checkpoint_path=checkpoint_path,
                trust_remote_code=trust_remote_code
            )
        else:
            model_path = model_name_or_path
        
        # 5. Create Workers
        console.print(
            f"  Creating {num_workers} vLLM Workers...",
            style=subhead_style_2,
        )
        self.workers = []
        
        vllm_kwargs = {
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "max_logprobs": max_logprobs,
        }

        if max_model_len is not None:
            vllm_kwargs["max_model_len"] = max_model_len

        vllm_kwargs.update(kwargs)
        
        # Determine whether to enable optimizations (default behavior, can be overridden by force flags)
        # Check if any task in task_types requires disabling optimizations
        enable_optimizations = self._should_enable_optimizations()
        
        # Create Ray remote class with dynamic GPU count and scheduling strategy
        # Use scheduling_strategy to place workers on specific nodes
        for i, (gpu_group, node_id) in enumerate(zip(self.worker_gpu_groups, self.worker_node_assignments)):
            # Create worker with node placement constraint
            VllmWorkerRemote = ray.remote(num_gpus=tensor_parallel_size)(VllmWorker)
            
            # If we have node assignment, use scheduling strategy
            if node_id is not None:
                from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
                scheduling_strategy = NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False  # Hard constraint: must be on this node
                )
                worker = VllmWorkerRemote.options(
                    scheduling_strategy=scheduling_strategy
                ).remote(
                    worker_id=i,
                    model_path=model_path,
                    gpu_ids=gpu_group,
                    enable_optimizations=enable_optimizations,
                    **vllm_kwargs
                )
            else:
                # No node constraint, let Ray decide
                worker = VllmWorkerRemote.remote(
                    worker_id=i,
                    model_path=model_path,
                    gpu_ids=gpu_group,
                    enable_optimizations=enable_optimizations,
                    **vllm_kwargs
                )
            self.workers.append(worker)
        
        # Wait for all Workers to initialize
        console.print(
            "  Waiting for all Workers to initialize...\n\n",
            style=subhead_style_2,
        )
        ray.get([worker.generate_batch.remote({}, {}) for worker in self.workers])
        
        console.print(
            f"✓ All Workers initialized successfully\n",
            style=success_style,
        )
        
        # Print optimization configuration summary
        if force_enable_optimizations:
            console.print(
                "⚙️ Optimization mode: FORCED ENABLED (chunked_prefill & prefix_caching enabled for all tasks)",
                style=warning_style,
            )
        elif force_disable_optimizations:
            console.print(
                "⚙️ Optimization mode: FORCED DISABLED (chunked_prefill & prefix_caching disabled for all tasks)",
                style=warning_style,
        )

        self.num_params = self._count_model_parameters()
        

    def _count_model_parameters(self) -> Optional[float]:
        """
        Override VllmMixin._count_model_parameters() for Ray-based generators.
        
        In Ray-based architecture, vLLM instances are in worker processes.
        Query the first worker to get model parameter count.
        
        Returns:
            float or None: Total number of parameters
        """
        tensor_parallel_size = getattr(self, 'tensor_parallel_size', 1)
        if tensor_parallel_size > 1:
            console.print(
                f"Warning: Tensor parallel (size={tensor_parallel_size}) detected. "
                f"Skipping parameter count (would only count local shard).",
                style=warning_style,
            )
            return None
        
        # Query the first worker to get parameter count
        try:
            import ray
            num_params = ray.get(self.workers[0].get_model_parameters.remote())
            console.print(
                f"✓ Model parameters: {num_params / 1e9:.2f}B\n",
                style=success_style,
            )
            return num_params
        except Exception as e:
            console.print(
                f"Warning: Failed to get parameter count from worker: {e}",
                style=warning_style,
            )
            return None

    def _generate_standard(
        self,
        prompts: Dict[str, str],
        **kwargs
    ) -> tuple:
        """
        Standard single-stage generation (round-robin assignment to multiple Workers)

        Args:
            prompts: {sample_id: prompt_text}
            **kwargs: Optional generation parameters, including:
                - worker_batch_size: Worker internal batch size, for avoiding vLLM scheduler issues (default 16)
                - return_logprobs: Whether to return cumulative logprobs for sampling mode (default False)

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [generated_text_1, generated_text_2, ...]}
            - Second dict: {sample_id: [cum_logprob_1, cum_logprob_2, ...]} (for beam search or when return_logprobs=True)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}} (lists for multi-stage support)
        """
        # Auto-enable return_logprobs if prompt_token is used (for recommendation tasks)
        has_prompt_token = bool(kwargs.get("prompt_token", None))
        
        # Build sampling parameters using mixin method
        sampling_params_obj = self._build_sampling_params(**kwargs)
        
        # Convert to dict for passing to workers
        if hasattr(sampling_params_obj, 'beam_width'):
            # BeamSearchParams
            use_beam_search = True
            sampling_params = {
                "use_beam_search": True,
                "beam_width": sampling_params_obj.beam_width,
                "max_tokens": sampling_params_obj.max_tokens,
            }
        else:
            # SamplingParams
            use_beam_search = False
            sampling_params = {
                "n": sampling_params_obj.n,
                "max_tokens": sampling_params_obj.max_tokens,
                "temperature": sampling_params_obj.temperature,
                "top_p": sampling_params_obj.top_p,
                "top_k": sampling_params_obj.top_k,
                "repetition_penalty": sampling_params_obj.repetition_penalty,
                "presence_penalty": sampling_params_obj.presence_penalty,
                "frequency_penalty": sampling_params_obj.frequency_penalty,
                "return_logprobs": kwargs.get("return_logprobs", has_prompt_token),
            }
            # Add stop parameter if specified
            if sampling_params_obj.stop:
                sampling_params["stop"] = sampling_params_obj.stop

        if use_beam_search:
            sampling_params["use_compact_beam_search"] = kwargs.get("use_compact_beam_search", False)

        console.print(
            f"Starting generation...",
            style=subhead_style_2,
        )
        if use_beam_search:
            console.print(
                f"Sampling parameters (beam search): beam_width={sampling_params['beam_width']}, "
                f"max_tokens={sampling_params['max_tokens']}",
                style=subhead_style_2,
            )
        else:
            console.print(
                f"Sampling parameters: n={sampling_params['n']}, max_tokens={sampling_params['max_tokens']}, "
                f"temperature={sampling_params['temperature']}, top_p={sampling_params['top_p']}, top_k={sampling_params['top_k']}, "
                f"repetition_penalty={sampling_params['repetition_penalty']}, "
                f"presence_penalty={sampling_params['presence_penalty']}, "
                f"frequency_penalty={sampling_params['frequency_penalty']}, "
                f"return_logprobs={sampling_params['return_logprobs']}",
                style=subhead_style_2,
            )
        
        # Round-robin assign tasks to Workers
        sample_ids = list(prompts.keys())
        num_workers = len(self.workers)
        worker_tasks = [dict() for _ in range(num_workers)]
        
        for i, sample_id in enumerate(sample_ids):
            worker_idx = i % num_workers
            worker_tasks[worker_idx][sample_id] = prompts[sample_id]
        
        console.print(
            f"Task distribution: {[len(task) for task in worker_tasks]}",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )
        
        # Execute in parallel
        futures = []
        for i, (worker, task) in enumerate(zip(self.workers, worker_tasks)):
            if task:  # Only submit non-empty tasks
                future = worker.generate_batch.remote(task, sampling_params, self.worker_batch_size)
                futures.append(future)
        
        # Collect results
        worker_results = ray.get(futures)

        # Merge results (each worker_result is a tuple of (texts_dict, logprobs_dict, mfu_stats_dict))
        results = {}
        logprobs = {}
        mfu_stats = {}
        for worker_result in worker_results:
            texts_dict, logprobs_dict, mfu_stats_dict = worker_result
            results.update(texts_dict)
            logprobs.update(logprobs_dict)
            mfu_stats.update(mfu_stats_dict)

        console.print(
            f"✓ Generation completed",
            style=success_style,
        )

        return (results, logprobs, mfu_stats)

    
    def extract_token_logprobs(
        self,
        prompts: Dict[str, str],
        target_tokens: List[str],
        **kwargs
    ) -> tuple:
        """
        Extract logprobs for specific target tokens (round-robin assignment to multiple Workers)

        Args:
            prompts: {sample_id: prompt_text}
            target_tokens: List of target tokens to extract probabilities for (e.g., ["是", "否"])
            **kwargs: Optional parameters including generation config

        Returns:
            Tuple of three dicts:
            - First dict: {sample_id: [json_string]} where json_string is formatted probabilities
            - Second dict: {} (empty, no beam search logprobs for classification)
            - Third dict: {sample_id: {"input_tokens": [int], "output_tokens": [int], "times": [float]}}
        """
        console.print(
            f"Extracting logprobs for tokens: {target_tokens}",
            style=subhead_style_2,
        )
        console.print(
            f"Worker batch size: {self.worker_batch_size}",
            style=subhead_style_2,
        )

        if not prompts:
            return ({}, {}, {})
        
        # Build sampling parameters
        sampling_params = {
            "n": kwargs.get("num_return_sequences", 1),
            "max_tokens": kwargs.get("max_new_tokens", 1),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", -1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "logprobs": kwargs.get("logprobs", 10),
        }
        
        console.print(
            f"Sampling parameters: n={sampling_params['n']}, max_tokens={sampling_params['max_tokens']}, "
            f"temperature={sampling_params['temperature']}, top_p={sampling_params['top_p']}, "
            f"top_k={sampling_params['top_k']}, repetition_penalty={sampling_params['repetition_penalty']}, "
            f"presence_penalty={sampling_params['presence_penalty']}, frequency_penalty={sampling_params['frequency_penalty']}, "
            f"logprobs={sampling_params['logprobs']}",
            style=subhead_style_2,
        )
        
        # Round-robin assign tasks to Workers
        sample_ids = list(prompts.keys())
        num_workers = len(self.workers)
        worker_tasks = [dict() for _ in range(num_workers)]
        
        for i, sample_id in enumerate(sample_ids):
            worker_idx = i % num_workers
            worker_tasks[worker_idx][sample_id] = prompts[sample_id]
        
        console.print(
            f"Task distribution: {[len(task) for task in worker_tasks]}",
            style=subhead_style_2,
        )

        # Get Worker internal batch size
        worker_batch_size = self.worker_batch_size

        # Execute in parallel
        futures = []
        for worker, task in zip(self.workers, worker_tasks):
            if task:  # Only submit non-empty tasks
                future = worker.extract_token_logprobs_batch.remote(
                    task, target_tokens, sampling_params, worker_batch_size
                )
                futures.append(future)
        
        # Collect results
        worker_results = ray.get(futures)

        # Merge results (each worker_result is a tuple of (probs_dict, mfu_stats_dict))
        results = {}
        mfu_stats = {}
        for worker_result in worker_results:
            probs_dict, mfu_stats_dict = worker_result
            results.update(probs_dict)
            mfu_stats.update(mfu_stats_dict)

        console.print(
            f"✓ Logprobs extraction completed",
            style=success_style,
        )

        return (results, {}, mfu_stats)
    
