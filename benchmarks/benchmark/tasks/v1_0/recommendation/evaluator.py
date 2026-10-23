"""
Recommendation Task Evaluator

Universal evaluator for all recommendation tasks.
Computes Pass@k and Position1_Pass@k metrics.
"""

import json
from typing import Dict, Any, Tuple, List

from benchmark.console import console, warning_style
from benchmark.tasks.v1_0.base_evaluator import BaseEval
from benchmark.tasks.v1_0.recommendation import utils as utils_sid
from benchmark.tasks.v1_0.recommendation import utils_by_pid as utils_pid
from benchmark.tasks.v1_0.recommendation.collision_ranking import rank_pid_candidates


class RecommendationEvaluator(BaseEval):
    """
    Universal evaluator for recommendation tasks

    Supports:
    - label_cond: Predict next video given specified consumption behavior
    - video: Next video prediction
    - product: Predict next clicked product
    - ad: Predict next clicked advertisement

    Metrics:
    - Pass@k: Check if any of top-k predictions match any ground truth SID
    - Position1_Pass@k: Check if any of top-k predictions match the first ground truth SID
    """

    @property
    def required_metrics(self) -> List[str]:
        """Define required overall metrics for Recommendation evaluation"""
        k_values = self.task_config.get("evaluation_config", {}).get("k_values", [128])
        evaluation_mode = self.task_config.get("evaluation_config", {}).get("evaluation_mode", "sid")

        metrics = []

        if evaluation_mode in ("sid", "both"):
            for k in k_values:
                metrics.extend([f"pass@{k}", f"position1_pass@{k}", f"recall@{k}"])

        if evaluation_mode in ("pid", "both"):
            for k in k_values:
                metrics.extend([f"pid_pass@{k}", f"pid_position1_pass@{k}", f"pid_recall@{k}"])

        return metrics

    def _has_all_required_metrics(self):
        config = self.task_config.get("evaluation_config", {})
        selected = config.get("select_k", "first_k")
        previous = self.cached_metrics.get("select_k_strategy")
        if selected == "pid_probability" or previous == "pid_probability":
            if previous != selected:
                return False
            if self.cached_metrics.get("pid_probability_weight") != config.get("pid_probability_weight", 1.0):
                return False
            if self.cached_metrics.get("sid_to_pid_strategy") != config.get("sid_to_pid_strategy", "most_popular_after_downsampling"):
                return False
        return super()._has_all_required_metrics()

    def _select_generations_by_strategy(
        self,
        generations: List[str],
        logprobs: List[float],
        strategy: str,
        code_to_pid=None,
        pid_probability_weight: float = 1.0
    ) -> List[str]:
        """
        Select and reorder generations based on the specified strategy

        Args:
            generations: List of generation strings
            logprobs: List of cumulative logprobs for each generation
            strategy: 'first_k', 'top_k_by_logprobs', or 'pid_probability'
            code_to_pid: Catalog required for collision-aware PID ranking
            pid_probability_weight: Nonnegative weight on the empirical PID share

        Returns:
            Reordered list of generations

        Raises:
            ValueError: If strategy is 'top_k_by_logprobs' but logprobs data is invalid
        """
        if strategy == "pid_probability":
            if code_to_pid is None:
                raise ValueError("PID probability ranking requires a SID-to-PID catalog")
            ranked = rank_pid_candidates(generations, logprobs, code_to_pid, pid_probability_weight)
            return [generations[candidate.index] for candidate in ranked]
        if strategy == "first_k":
            # Keep original order
            return generations
        elif strategy == "top_k_by_logprobs":
            # Validate logprobs data
            if not logprobs:
                raise ValueError(
                    f"Strategy 'top_k_by_logprobs' requires logprobs data, but logprobs is empty. "
                    f"Please ensure the generation was run with logprobs enabled."
                )
            if len(logprobs) != len(generations):
                raise ValueError(
                    f"Strategy 'top_k_by_logprobs' requires logprobs length to match generations length. "
                    f"Got logprobs length {len(logprobs)}, generations length {len(generations)}."
                )

            # Sort generations by logprobs in descending order (higher logprob = better)
            paired = list(zip(generations, logprobs))
            paired_sorted = sorted(paired, key=lambda x: x[1], reverse=True)

            # Deduplicate while preserving order (keep first occurrence with highest logprob)
            seen = set()
            unique_generations = []
            for gen, _ in paired_sorted:
                if gen not in seen:
                    seen.add(gen)
                    unique_generations.append(gen)

            return unique_generations
        else:
            raise ValueError(
                f"Unknown selection strategy: '{strategy}'. "
                f"Supported strategies: 'first_k', 'top_k_by_logprobs', 'pid_probability'"
            )

    def _evaluate_single_mode(
        self,
        k_values: List[int],
        evaluation_mode: str,
        select_k_strategy: str,
        code_to_pid: Dict[int, List[Tuple[int, float]]] = None,
        sid_to_pid_strategy: str = "most_popular"
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float], Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
        """
        Evaluate samples using a single mode (SID or PID)

        Args:
            k_values: List of k values to compute
            evaluation_mode: Either 'sid' or 'pid'
            select_k_strategy: Selection strategy for generations
            code_to_pid: PID mapping dictionary (required for 'pid' mode)
            sid_to_pid_strategy: Strategy for SID->PID conversion ("most_popular" or "random")

        Returns:
            Tuple of (pass_counts, position1_pass_counts, recall_sums, per_sample_metrics, debug_info_lists)
        """
        # Select utils module based on mode
        if evaluation_mode == "sid":
            utils = utils_sid
        elif evaluation_mode == "pid":
            if code_to_pid is None:
                raise ValueError("code_to_pid is required for PID evaluation mode")
            utils = utils_pid
        else:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")

        # Initialize counters
        pass_at_k_counts = {k: 0 for k in k_values}
        position1_pass_at_k_counts = {k: 0 for k in k_values}
        recall_at_k_sums = {k: 0.0 for k in k_values}

        # Per-sample metrics collection
        per_sample_metrics = {}

        # Debug information collection
        debug_info = {
            "passed_samples": [],
            "failed_samples": [],
            "no_generation_samples": [],
        }

        # Helper function to create failed metrics
        def create_failed_metrics():
            """Create metrics dict for failed samples (all False/0.0)"""
            metrics = {}
            for k in k_values:
                metrics[f"pass@{k}"] = False
                metrics[f"position1_pass@{k}"] = False
                metrics[f"recall@{k}"] = 0.0
            return metrics

        for sample_id, sample in self.samples.items():
            # Get model predictions
            generations = sample.get("generations", [])
            logprobs = sample.get("logprobs", [])

            if not generations:
                # No generation, treat as failure
                per_sample_metrics[sample_id] = create_failed_metrics()

                if self.debug:
                    debug_info["no_generation_samples"].append({
                        "sample_id": sample_id,
                        "ground_truth": sample.get("ground_truth", "") if evaluation_mode == "sid" else sample.get("metadata", {}).get("answer_pid", []),
                    })
                continue

            # Get ground truth based on mode
            if evaluation_mode == "sid":
                ground_truth = sample.get("ground_truth", "")
                ground_truth_ids = utils.extract_ids_from_answer(ground_truth)
                first_ground_truth_id = utils.extract_first_id_from_answer(ground_truth)
            else:  # pid mode
                # Try answer_pid first, fallback to answer_iid if not available
                ground_truth_pids = sample.get("metadata", {}).get("answer_pid")
                if ground_truth_pids is None:
                    ground_truth_pids = sample.get("metadata", {}).get("answer_iid", [])
                if isinstance(ground_truth_pids, str):
                    ground_truth_pids = json.loads(ground_truth_pids)
                ground_truth_ids = utils.extract_ids_from_answer(ground_truth_pids)
                first_ground_truth_id = utils.extract_first_id_from_answer(ground_truth_pids)

            if not ground_truth_ids:
                console.print(f"Sample {sample_id}: no valid ID found in ground truth ({evaluation_mode} mode)", style=warning_style)
                per_sample_metrics[sample_id] = create_failed_metrics()
                continue

            # Apply selection strategy to reorder generations
            selected_generations = self._select_generations_by_strategy(
                generations=generations,
                logprobs=logprobs,
                strategy=select_k_strategy,
                code_to_pid=code_to_pid,
                pid_probability_weight=self.task_config.get("evaluation_config", {}).get("pid_probability_weight", 1.0)
            )

            # Extract predicted IDs from selected generations
            if evaluation_mode == "sid":
                predicted_ids = [utils.extract_id_from_generation(gen) for gen in selected_generations]
            else:  # pid mode
                predicted_ids = [utils.extract_id_from_generation(gen, code_to_pid, sid_to_pid_strategy) for gen in selected_generations]

            # Compute metrics for each k
            sample_pass_results = {}
            sample_position1_pass_results = {}
            sample_recall_results = {}

            for k in k_values:
                # Compute Pass@k
                pass_result = utils.compute_pass_at_k(predicted_ids, ground_truth_ids, k)
                sample_pass_results[f"pass@{k}"] = pass_result
                if pass_result:
                    pass_at_k_counts[k] += 1

                # Compute Position1_Pass@k
                position1_pass_result = utils.compute_position1_pass_at_k(
                    predicted_ids, first_ground_truth_id, k
                )
                sample_position1_pass_results[f"position1_pass@{k}"] = position1_pass_result
                if position1_pass_result:
                    position1_pass_at_k_counts[k] += 1

                # Compute Recall@k
                recall_result = utils.compute_recall_at_k(predicted_ids, ground_truth_ids, k)
                sample_recall_results[f"recall@{k}"] = recall_result
                recall_at_k_sums[k] += recall_result

            # Store per-sample metrics
            sample_metrics = {
                **sample_pass_results,
                **sample_position1_pass_results,
                **sample_recall_results
            }

            # For PID mode, save pid_generations (convert None/invalid to -1)
            if evaluation_mode == "pid":
                pid_generations = [pid if pid is not None else -1 for pid in predicted_ids]
                sample_metrics["generations"] = pid_generations

            per_sample_metrics[sample_id] = sample_metrics

            # Debug information collection
            if self.debug:
                metadata = sample.get("metadata", {})
                raw_prompt = metadata.get("raw_prompt", "")

                if evaluation_mode == "sid":
                    debug_item = utils.get_debug_info(
                        sample_id=sample_id,
                        generations=generations,
                        ground_truth=sample.get("ground_truth", ""),
                        pass_results=sample_pass_results,
                        position1_pass_results=sample_position1_pass_results,
                        raw_prompt=raw_prompt,
                    )
                else:  # pid mode
                    answer_pid = metadata.get("answer_pid", metadata.get("answer_iid", []))
                    if isinstance(answer_pid, str):
                        answer_pid = json.loads(answer_pid)
                    debug_item = utils.get_debug_info(
                        sample_id=sample_id,
                        generations=generations,
                        ground_truth=answer_pid,
                        pass_results=sample_pass_results,
                        position1_pass_results=sample_position1_pass_results,
                        code_to_pid=code_to_pid,
                        strategy=sid_to_pid_strategy,
                        raw_prompt=raw_prompt,
                    )

                # Check if any pass@k is True
                if any(sample_pass_results.values()):
                    debug_info["passed_samples"].append(debug_item)
                else:
                    debug_info["failed_samples"].append(debug_item)

        return pass_at_k_counts, position1_pass_at_k_counts, recall_at_k_sums, per_sample_metrics, debug_info

    def _calculate_metrics_from_counts(
        self,
        pass_counts: Dict[int, int],
        position1_pass_counts: Dict[int, int],
        recall_sums: Dict[int, float],
        total_samples: int,
        k_values: List[int],
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate metrics from counts

        Args:
            pass_counts: Pass@k counts for each k
            position1_pass_counts: Position1_Pass@k counts for each k
            recall_sums: Recall@k sums for each k
            total_samples: Total number of samples
            k_values: List of k values
            prefix: Prefix for metric names (e.g., "pid_")

        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        for k in k_values:
            metrics[f"{prefix}pass@{k}"] = pass_counts[k] / total_samples if total_samples > 0 else 0.0
            metrics[f"{prefix}position1_pass@{k}"] = position1_pass_counts[k] / total_samples if total_samples > 0 else 0.0
            metrics[f"{prefix}recall@{k}"] = recall_sums[k] / total_samples if total_samples > 0 else 0.0
        return metrics

    def _compute_metrics_from_scratch(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Compute all evaluation metrics from scratch

        Returns:
            Tuple of (metrics, per_sample_metrics)
        """
        total_samples = len(self.samples)

        # Get configuration
        evaluation_config = self.task_config.get('evaluation_config', {})
        k_values = evaluation_config.get("k_values", [128])
        select_k_strategy = evaluation_config.get('select_k', 'first_k')
        evaluation_mode = evaluation_config.get('evaluation_mode', 'both')
        sid_to_pid_strategy = evaluation_config.get('sid_to_pid_strategy', 'most_popular_after_downsampling')

        if select_k_strategy == "pid_probability":
            if evaluation_mode not in ("pid", "both"):
                raise ValueError("pid_probability requires PID or both evaluation modes")
            if sid_to_pid_strategy != "most_popular_after_downsampling_deterministic":
                raise ValueError("pid_probability requires most_popular_after_downsampling_deterministic resolution")

        # Load PID mapping if needed
        code_to_pid = None
        if evaluation_mode in ("pid", "both"):
            from pathlib import Path
            task_name = self.task_config.get("name", "")
            if task_name == "product":
                mapping_filename = "sid2iid.json"
            else:
                mapping_filename = "sid2pid.json"
            pid_mapping_path = str(Path(self.data_dir) / mapping_filename)
            console.print(f"[cyan]Loading PID mapping from {pid_mapping_path}...[/cyan]")
            code_to_pid = utils_pid.load_pid_mapping(pid_mapping_path)

        # Define evaluation modes to run
        # Format: (mode_name, metric_prefix, debug_filename, log_message)
        modes_config = {
            "sid": [("sid", "", "debug.json", "Evaluating using SID mode...")],
            "pid": [("pid", "pid_", "debug_pid.json", "Evaluating using PID mode...")],
            "both": [
                ("sid", "", "debug_sid.json", "  Running SID evaluation..."),
                ("pid", "pid_", "debug_pid.json", "  Running PID evaluation...")
            ]
        }

        if evaluation_mode not in modes_config:
            raise ValueError(f"Invalid evaluation_mode: '{evaluation_mode}'. Must be 'sid', 'pid', or 'both'")

        if evaluation_mode == "both":
            console.print("[cyan]Evaluating using both SID and PID modes...[/cyan]")

        # Initialize metrics
        metrics = {"total_samples": total_samples}
        per_sample_metrics = {}
        all_debug_info = {}

        # Run evaluation for each configured mode
        for mode_name, metric_prefix, debug_filename, log_message in modes_config[evaluation_mode]:
            console.print(f"[cyan]{log_message}[/cyan]")

            # Run evaluation
            pass_counts, position1_pass_counts, recall_sums, mode_per_sample_metrics, debug_info = self._evaluate_single_mode(
                k_values=k_values,
                evaluation_mode=mode_name,
                select_k_strategy=select_k_strategy,
                code_to_pid=code_to_pid if mode_name == "pid" or select_k_strategy == "pid_probability" else None,
                sid_to_pid_strategy=sid_to_pid_strategy if mode_name == "pid" else "most_popular"
            )

            # Calculate and add metrics
            mode_metrics = self._calculate_metrics_from_counts(
                pass_counts, position1_pass_counts, recall_sums,
                total_samples, k_values, metric_prefix
            )
            metrics.update(mode_metrics)

            # Merge per-sample metrics with appropriate prefix
            for sample_id, sample_metric in mode_per_sample_metrics.items():
                if sample_id not in per_sample_metrics:
                    per_sample_metrics[sample_id] = {}
                # Add metrics with prefix (for PID mode) or without (for SID mode)
                if metric_prefix:
                    # PID mode: add prefix to metric names
                    for metric_name, metric_value in sample_metric.items():
                        prefixed_name = f"{metric_prefix}{metric_name}"
                        per_sample_metrics[sample_id][prefixed_name] = metric_value
                else:
                    # SID mode: no prefix
                    per_sample_metrics[sample_id].update(sample_metric)

            # Store debug info for later saving
            if self.debug and self.predictions_dir:
                all_debug_info[mode_name] = (debug_info, debug_filename, mode_metrics)

        # Save debug info
        if self.debug and self.predictions_dir:
            for mode_name, (debug_info, debug_filename, mode_metrics) in all_debug_info.items():
                # For single mode, include all metrics; for both mode, filter by prefix
                if evaluation_mode == "both":
                    prefix = "pid_" if mode_name == "pid" else ""
                    filtered_metrics = {
                        k: v for k, v in mode_metrics.items()
                        if k == "total_samples" or k.startswith(prefix)
                    }
                    filtered_metrics["total_samples"] = total_samples
                else:
                    filtered_metrics = dict(metrics)

                self._save_debug_info(debug_info, filtered_metrics, debug_filename)

        # Record configuration
        metrics["select_k_strategy"] = select_k_strategy
        metrics["evaluation_mode"] = evaluation_mode
        if select_k_strategy == "pid_probability":
            metrics["pid_probability_weight"] = evaluation_config.get("pid_probability_weight", 1.0)
        if evaluation_mode in ("pid", "both"):
            metrics["sid_to_pid_strategy"] = sid_to_pid_strategy

        return metrics, per_sample_metrics

    def _save_debug_info(self, debug_info: Dict[str, Any], metrics: Dict[str, Any], debug_filename: str = None):
        """
        Save detailed debug information to file

        Args:
            debug_info: Debug information dictionary
            metrics: Overall metrics
            debug_filename: Optional custom filename (absolute path or relative to predictions_dir)
        """
        # Add statistics to debug_info
        debug_info["statistics"] = {
            "total_samples": metrics.get("total_samples", 0),
            "passed_samples_count": len(debug_info.get("passed_samples", [])),
            "failed_samples_count": len(debug_info.get("failed_samples", [])),
            "no_generation_samples_count": len(debug_info.get("no_generation_samples", [])),
        }

        # Add metrics
        debug_info["metrics"] = metrics

        # Use default filename if not specified
        if debug_filename is None:
            debug_filename = "debug.json"

        # Save debug info to file using base class method
        self._save_debug_json(debug_info, filename=debug_filename)

        console.print(f"Total samples: {metrics['total_samples']}")
        console.print(f"Passed samples: {len(debug_info['passed_samples'])}")
        console.print(f"Failed samples: {len(debug_info['failed_samples'])}")
        console.print(f"No generation samples: {len(debug_info['no_generation_samples'])}")

        # Print metrics
        console.print("\n[bold]Metrics:[/bold]")
        for metric_name, metric_value in metrics.items():
            if metric_name != "total_samples":
                console.print(f"  {metric_name}: {metric_value}")

        # Show some failed examples
        if debug_info["failed_samples"]:
            console.print(f"\n[yellow]Failed sample examples (first 3):[/yellow]")
            for i, item in enumerate(debug_info["failed_samples"][:3]):
                console.print(f"  Example {i+1}:")
                console.print(f"    Sample ID: {item['sample_id']}")
                # Handle both SID and PID modes
                if 'ground_truth_sids' in item:
                    console.print(f"    Ground truth SIDs: {item['ground_truth_sids']}")
                elif 'ground_truth_pids' in item:
                    console.print(f"    Ground truth PIDs: {item['ground_truth_pids']}")
                console.print(f"    Top 5 generations: {item['top_10_generations'][:5]}")
                console.print()

