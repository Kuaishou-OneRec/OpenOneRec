"""
Recommendation Task Utilities (PID-based)

Functions for PID extraction and recommendation metrics computation using PIDs.
"""

import re
import json
import random
from typing import Set, Dict, List, Any, Tuple, Optional
from pathlib import Path
from collections import Counter


# Encoding constants for (code1, code2, code3) -> single int
# Each code is in range [0, 8192], needs 13 bits
CODE_MULTIPLIER_1 = 8192 * 8192  # 67108864
CODE_MULTIPLIER_2 = 8192


def load_pid_mapping(mapping_path: str) -> Dict[int, List[Dict[str, int]]]:
    """
    Load SID to PID mapping from JSON file

    Args:
        mapping_path: Path to the JSON file containing SID to PID mapping

    Returns:
        Dictionary mapping encoded SID (int) to list of PID info dictionaries
        Format: {encoded_sid: [{"pid": pid1, "count": count1, "count_after_downsample": count2}, ...]}
        PIDs are sorted by original count in descending order
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"PID mapping file not found: {mapping_path}")

    with open(mapping_path, 'r') as f:
        sid_to_pid_json = json.load(f)
    
    # Convert string keys back to integers
    code_to_pid = {int(k): v for k, v in sid_to_pid_json.items()}

    print(f"[INFO] Loaded {len(code_to_pid)} SID to PID mappings from {mapping_path}")
    return code_to_pid


def encode_sid(c1: int, c2: int, c3: int) -> int:
    """
    Encode (code1, code2, code3) into a single integer key

    Args:
        c1, c2, c3: SID code components

    Returns:
        Encoded integer key
    """
    return c1 * CODE_MULTIPLIER_1 + c2 * CODE_MULTIPLIER_2 + c3


def extract_sid_codes_from_text(text: str) -> Optional[Tuple[int, int, int]]:
    """
    Extract SID codes from text using regex pattern

    Args:
        text: Input text containing SID patterns like <|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>

    Returns:
        Tuple (a, b, c) representing extracted SID codes, or None if not found
        Expects exactly one SID in the text
    """
    pattern = r'<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>'
    matches = re.findall(pattern, text)
    if not matches:
        return None
    if len(matches) > 1:
        # Log warning but use first match
        print(f"[WARNING] Expected 1 SID code, got {len(matches)}, using first")
    return (int(matches[0][0]), int(matches[0][1]), int(matches[0][2]))


def _get_id_from_info(info: Dict[str, int]) -> int:
    """
    Extract ID from info dict, supporting both 'pid' and 'iid' keys.

    Args:
        info: Dictionary containing either 'pid' or 'iid' key

    Returns:
        The ID value (int)
    """
    return info.get("pid", info.get("iid", 0))


def apply_sid_to_pid_strategy(pid_info_list: List[Dict[str, int]], strategy: str) -> int:
    """
    Apply strategy to select a single PID from a list

    Args:
        pid_info_list: List of PID info dictionaries
                      Format: [{"pid": pid1, "count": count1, "count_after_downsample": count2}, ...]
                      or [{"iid": iid1, "count": count1, "count_after_downsample": count2}, ...] for product
        strategy: One of "most_popular_originally", "most_popular_after_downsampling", or "random"

    Returns:
        Selected PID/IID (int), or 0 if list is empty

    Strategies:
        - "most_popular_originally": Return the PID with highest original count (already sorted)
        - "most_popular_after_downsampling": Return the PID with highest downsampled count (random if tie)
        - "random": Randomly select one PID from the list
    """
    if not pid_info_list:
        return 0

    if strategy == "most_popular_originally":
        # Return the first PID/IID (highest original count, already sorted)
        return _get_id_from_info(pid_info_list[0])
    elif strategy == "most_popular_after_downsampling":
        # Find max downsampled count
        max_count = max(info["count_after_downsample"] for info in pid_info_list)
        # Get all PIDs/IIDs with max downsampled count
        max_pids = [_get_id_from_info(info) for info in pid_info_list if info["count_after_downsample"] == max_count]
        # Randomly select one if there are ties
        return random.choice(max_pids)
    elif strategy == "random":
        # Randomly select a PID/IID
        return random.choice([_get_id_from_info(info) for info in pid_info_list])
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Must be 'most_popular_originally', 'most_popular_after_downsampling', or 'random'")


def extract_ids_from_answer(answer: list[int]) -> list[int]:
    """Extract all PIDs from answer field, preserving original order.

    Returns a deduplicated list that keeps the first occurrence order.

    >>> extract_ids_from_answer([123, 456, 123, 789])
    [123, 456, 789]
    """
    seen: set[int] = set()
    correct_answers: list[int] = []
    for pid in answer:
        if pid != 0 and pid not in seen:
            correct_answers.append(pid)
            seen.add(pid)
    return correct_answers


def extract_first_id_from_answer(answer: List[int]) -> int:
    """
    Extract the first PID from answer field

    Examples:
        >>> extract_first_id_from_answer([123, 456, 789])
        123
    """
    valid_pids = [pid for pid in answer if pid != 0]
    return valid_pids[0] if valid_pids else 0


def extract_id_from_generation(
    generation: str,
    code_to_pid: Dict[int, List[Dict[str, int]]],
    strategy: str = "most_popular_originally"
) -> int:
    """
    Extract PID from model generation

    The generation may contain:
    - SID wrapped in tags: "<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>"
    - With thinking: "<think>...</think>\\n<|sid_begin|><s_a_1>..."

    Args:
        generation: Model generation string (contains exactly one SID)
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")

    Returns:
        Extracted PID (int), or 0 if not found

    Examples:
        >>> extract_id_from_generation("<|sid_begin|><s_a_1><s_b_2><s_c_3><|sid_end|>", code_to_pid)
        12345  # Assuming this SID maps to PID 12345
    """
    generation = generation.strip()

    # If generation contains </think>, only process content after it
    if '</think>' in generation:
        generation = generation.split('</think>')[-1].strip()

    # Extract SID codes from the generation (should be exactly one)
    sid_codes = extract_sid_codes_from_text(generation)

    if sid_codes is None:
        return 0

    # Encode SID and look up PID list
    encoded = encode_sid(*sid_codes)
    pid_freq_list = code_to_pid.get(encoded, [])

    # Apply strategy to select PID
    return apply_sid_to_pid_strategy(pid_freq_list, strategy)


def compute_pass_at_k(
    predicted_ids: List[int],
    ground_truth_ids: Set[int],
    k: int
) -> bool:
    """
    Compute Pass@k for a single sample using PIDs

    Pass@k definition:
    - Take the first k candidate PIDs from predictions
    - If any of these k PIDs appears in the ground truth PIDs, return True

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        ground_truth_ids: Set of ground truth PIDs
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match ground truth, False otherwise
    """
    if not predicted_ids or not ground_truth_ids:
        return False

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Check if any matches ground truth
    for pid in top_k_ids:
        if pid != 0 and pid in ground_truth_ids:
            return True

    return False


def compute_position1_pass_at_k(
    predicted_ids: List[int],
    first_ground_truth_id: int,
    k: int
) -> bool:
    """
    Compute Position1_Pass@k for a single sample using PIDs

    Position1_Pass@k definition:
    - Take the first k candidate PIDs from predictions
    - Only consider the first PID in the ground truth
    - If any of these k PIDs matches the first ground truth, return True

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        first_ground_truth_id: The first ground truth PID
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match the first ground truth, False otherwise
    """
    if not predicted_ids or not first_ground_truth_id or first_ground_truth_id == 0:
        return False

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Check if any matches the first ground truth
    for pid in top_k_ids:
        if pid != 0 and pid == first_ground_truth_id:
            return True

    return False


def compute_recall_at_k(
    predicted_ids: List[int],
    ground_truth_ids: Set[int],
    k: int
) -> float:
    """
    Compute Recall@k for a single sample using PIDs

    Recall@k definition:
    - Take the first k candidate PIDs from predictions
    - Count how many unique ground truth PIDs are hit by these k PIDs
    - Return the ratio: hit_count / total_ground_truth_count

    Args:
        predicted_ids: List of predicted PIDs (already extracted from generations)
        ground_truth_ids: Set of ground truth PIDs
        k: Number of top predictions to consider

    Returns:
        Recall@k score (0.0 to 1.0)

    Examples:
        >>> predicted_ids = [123, 456, 999, 888]
        >>> ground_truth_ids = {123, 456, 789}
        >>> compute_recall_at_k(predicted_ids, ground_truth_ids, k=2)
        0.6667  # Hit 2 out of 3 ground truth PIDs
    """
    if not predicted_ids or not ground_truth_ids:
        return 0.0

    # Take first k predicted PIDs
    top_k_ids = predicted_ids[:k]

    # Convert to set and filter out zeros
    predicted_ids_set = set(pid for pid in top_k_ids if pid != 0)

    # Count how many ground truth PIDs are hit
    hit_count = len(predicted_ids_set & ground_truth_ids)  # Set intersection

    # Calculate recall
    recall = hit_count / len(ground_truth_ids)

    return recall


def get_unique_generations(
    generations: List[str],
    max_count: int,
    code_to_pid: Dict[int, List[Dict[str, int]]],
    strategy: str = "most_popular_originally",
    logprobs: List[float] = None,
    exclude_ids: Set[int] = None,
    sources: List[str] = None
):
    """
    Get first N unique PIDs from generations, optionally sorted by logprobs

    This function extracts unique PIDs, optionally sorting by logprobs first.
    Useful for merging results from multiple generation runs.

    Args:
        generations: List of model generation strings containing SID patterns
        max_count: Maximum number of unique PIDs to return
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")
        logprobs: Optional list of log probabilities (same length as generations)
        exclude_ids: Optional set of PIDs to exclude from results
        sources: Optional list of source labels (same length as generations)

    Returns:
        List of unique PIDs (up to max_count), sorted by logprobs if provided
        If sources provided, returns tuple (List[int], List[str]) of (unique_pids, corresponding_sources)
    """
    # Track sources if provided
    track_sources = sources is not None and len(sources) == len(generations)

    # If logprobs provided, sort generations by logprobs (descending)
    if logprobs is not None and len(logprobs) == len(generations):
        # Create tuples and sort by logprob (descending)
        if track_sources:
            gen_data = list(zip(generations, logprobs, sources))
            gen_data.sort(key=lambda x: x[1], reverse=True)
            sorted_generations = [gen for gen, _, _ in gen_data]
            sorted_sources = [src for _, _, src in gen_data]
        else:
            gen_logprob_pairs = list(zip(generations, logprobs))
            gen_logprob_pairs.sort(key=lambda x: x[1], reverse=True)
            sorted_generations = [gen for gen, _ in gen_logprob_pairs]
            sorted_sources = None
    else:
        sorted_generations = generations
        sorted_sources = sources if track_sources else None

    seen = set()
    unique_pids = []
    unique_sources = [] if track_sources else None
    exclude = exclude_ids or set()

    for i, gen in enumerate(sorted_generations):
        # Skip empty strings
        if not gen or not gen.strip():
            continue

        # Extract PID from generation text
        pid = extract_id_from_generation(gen, code_to_pid, strategy)

        # Skip if PID is 0 (not found), already seen, or in exclude list
        if pid == 0 or pid in seen or pid in exclude:
            continue

        unique_pids.append(pid)
        seen.add(pid)

        if track_sources:
            unique_sources.append(sorted_sources[i])

        # Stop if we've collected enough unique PIDs
        if len(unique_pids) >= max_count:
            break

    if track_sources:
        return unique_pids, unique_sources
    return unique_pids


def get_debug_info(
    sample_id: str,
    generations: List[str],
    ground_truth: List[int],
    pass_results: Dict[str, bool],
    position1_pass_results: Dict[str, bool],
    code_to_pid: Dict[int, List[Dict[str, int]]],
    strategy: str = "most_popular_originally",
    raw_prompt: str = ""
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample (PID-based)

    Args:
        sample_id: Sample ID
        generations: List of generated SIDs
        ground_truth: Ground truth answer string
        pass_results: Pass@k results for this sample
        position1_pass_results: Position1_Pass@k results for this sample
        code_to_pid: Mapping dictionary {encoded_sid: [{"pid": pid, "count": ..., "count_after_downsample": ...}, ...]}
        strategy: Strategy for selecting PID ("most_popular_originally", "most_popular_after_downsampling", or "random")
        raw_prompt: Raw prompt (optional)

    Returns:
        Debug information dictionary
    """
    ground_truth_ids = extract_ids_from_answer(ground_truth)
    first_ground_truth_id = extract_first_id_from_answer(ground_truth)

    # Extract top-k generated PIDs
    top_k_ids = [extract_id_from_generation(gen, code_to_pid, strategy) for gen in generations[:10]]

    debug_item = {
        "sample_id": sample_id,
        "ground_truth_pids": list(ground_truth_ids),
        "first_ground_truth_pid": first_ground_truth_id,
        "top_10_generations": top_k_ids,
        "pass_results": pass_results,
        "position1_pass_results": position1_pass_results,
    }

    if raw_prompt:
        debug_item["raw_prompt_snippet"] = raw_prompt[:200] + "..." if len(raw_prompt) > 200 else raw_prompt

    return debug_item
