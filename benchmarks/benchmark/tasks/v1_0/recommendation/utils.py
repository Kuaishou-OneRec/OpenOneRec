"""
Recommendation Task Utilities

Functions for SID extraction and recommendation metrics computation.
"""

from typing import Set, Dict, List, Any


def extract_ids_from_answer(answer: str) -> list[str]:
    """Extract all SIDs from answer field, preserving original order.

    Returns a deduplicated list that keeps the first occurrence order.

    >>> extract_ids_from_answer("<|sid_begin|>123<|sid_end|><|sid_begin|>456<|sid_end|>")
    ['123', '456']
    """
    seen: set[str] = set()
    correct_answers: list[str] = []
    for part in answer.split('<|sid_begin|>'):
        if '<|sid_end|>' in part:
            sid = part.split('<|sid_end|>')[0].strip()
            if sid and sid not in seen:
                correct_answers.append(sid)
                seen.add(sid)
    return correct_answers


def extract_first_id_from_answer(answer: str) -> str:
    """
    Extract the first SID from answer field
    
    Args:
        answer: String containing multiple <|sid_begin|>...<|sid_end|> patterns
    
    Returns:
        The first extracted SID, or empty string if none found
    
    Examples:
        >>> extract_first_id_from_answer("<|sid_begin|>123<|sid_end|><|sid_begin|>456<|sid_end|>")
        '123'
    """
    for part in answer.split('<|sid_begin|>'):
        if '<|sid_end|>' in part:
            sid = part.split('<|sid_end|>')[0].strip()
            if sid:
                return sid
    return ""


def extract_id_from_generation(generation: str) -> str:
    """
    Extract SID from model generation

    The generation may contain:
    - SID directly: "123"
    - Wrapped in tags: "<|sid_begin|>123<|sid_end|>"
    - With thinking: "<think>...</think>\\n<|sid_begin|>123" (two-stage generation)

    Args:
        generation: Model generation string

    Returns:
        Extracted SID, or the stripped generation if no pattern found

    Examples:
        >>> extract_id_from_generation("<|sid_begin|>123<|sid_end|>")
        '123'
        >>> extract_id_from_generation("123")
        '123'
        >>> extract_id_from_generation("<think>reasoning</think>\\n<|sid_begin|>123")
        '123'
    """
    generation = generation.strip()

    # If generation contains </think>, only process content after it
    if '</think>' in generation:
        generation = generation.split('</think>')[-1].strip()

    # Try to extract from <|sid_begin|>...<|sid_end|> pattern
    if '<|sid_begin|>' in generation:
        for part in generation.split('<|sid_begin|>'):
            if '<|sid_end|>' in part:
                sid = part.split('<|sid_end|>')[0].strip()
                if sid:
                    return sid
            elif part.strip():  # No end marker, take the content after begin marker
                return part.strip()

    # Otherwise, return the stripped generation
    return generation


def compute_pass_at_k(
    predicted_sids: List[str],
    ground_truth_sids: Set[str],
    k: int
) -> bool:
    """
    Compute Pass@k for a single sample

    Pass@k definition:
    - Take the first k candidate SIDs from predictions
    - If any of these k SIDs appears in the ground truth SIDs, return True

    Args:
        predicted_sids: List of predicted SIDs (already extracted from generations)
        ground_truth_sids: Set of ground truth SIDs
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match ground truth, False otherwise
    """
    if not predicted_sids or not ground_truth_sids:
        return False

    # Take first k predicted SIDs
    top_k_sids = predicted_sids[:k]

    # Check if any matches ground truth
    for sid in top_k_sids:
        if sid in ground_truth_sids:
            return True

    return False


def compute_position1_pass_at_k(
    predicted_sids: List[str],
    first_ground_truth_sid: str,
    k: int
) -> bool:
    """
    Compute Position1_Pass@k for a single sample

    Position1_Pass@k definition:
    - Take the first k candidate SIDs from predictions
    - Only consider the first SID in the ground truth
    - If any of these k SIDs matches the first ground truth, return True

    Args:
        predicted_sids: List of predicted SIDs (already extracted from generations)
        first_ground_truth_sid: The first ground truth SID
        k: Number of top predictions to consider

    Returns:
        True if any of the top-k predictions match the first ground truth, False otherwise
    """
    if not predicted_sids or not first_ground_truth_sid:
        return False

    # Take first k predicted SIDs
    top_k_sids = predicted_sids[:k]

    # Check if any matches the first ground truth
    for sid in top_k_sids:
        if sid == first_ground_truth_sid:
            return True

    return False


def compute_recall_at_k(
    predicted_sids: List[str],
    ground_truth_sids: Set[str],
    k: int
) -> float:
    """
    Compute Recall@k for a single sample

    Recall@k definition:
    - Take the first k candidate SIDs from predictions
    - Count how many unique ground truth SIDs are hit by these k SIDs
    - Return the ratio: hit_count / total_ground_truth_count

    Args:
        predicted_sids: List of predicted SIDs (already extracted from generations)
        ground_truth_sids: Set of ground truth SIDs
        k: Number of top predictions to consider

    Returns:
        Recall@k score (0.0 to 1.0)

    Examples:
        >>> predicted_sids = ["123", "456", "999", "888"]
        >>> ground_truth_sids = {"123", "456", "789"}
        >>> compute_recall_at_k(predicted_sids, ground_truth_sids, k=2)
        0.6667  # Hit 2 out of 3 ground truth SIDs
        >>> compute_recall_at_k(predicted_sids, ground_truth_sids, k=4)
        0.6667  # Still hit only 2, since 789 is not in top-4
    """
    if not predicted_sids or not ground_truth_sids:
        return 0.0

    # Take first k predicted SIDs
    top_k_sids = predicted_sids[:k]

    # Convert to set and filter out empty strings
    predicted_sids_set = set(sid for sid in top_k_sids if sid)

    # Count how many ground truth SIDs are hit
    hit_count = len(predicted_sids_set & ground_truth_sids)  # Set intersection

    # Calculate recall
    recall = hit_count / len(ground_truth_sids)

    return recall


def get_unique_generations(
    generations: List[str],
    max_count: int,
    logprobs: List[float] = None,
    exclude_sids: Set[str] = None,
    sources: List[str] = None
):
    """
    Get first N unique SIDs from generations, optionally sorted by logprobs

    This function extracts unique SIDs, optionally sorting by logprobs first.
    Useful for merging results from multiple generation runs.

    Args:
        generations: List of model generation strings (may contain <|sid_begin|>...<|sid_end|> or <think>...</think>)
        max_count: Maximum number of unique SIDs to return
        logprobs: Optional list of log probabilities (same length as generations). If provided, sorts by logprobs (descending) before extracting unique SIDs
        exclude_sids: Optional set of SIDs to exclude from results
        sources: Optional list of source labels (same length as generations). If provided, returns tuple (sids, sources)

    Returns:
        List of unique SIDs (up to max_count), sorted by logprobs if provided, otherwise in generation order
        If sources provided, returns tuple (List[str], List[str]) of (unique_sids, corresponding_sources)

    Examples:
        >>> gens = ["<|sid_begin|>123<|sid_end|>", "456", "<think>...</think>\\n123", "789", "456", "999"]
        >>> get_unique_generations(gens, max_count=3)
        ['123', '456', '789']
        >>> get_unique_generations(gens, max_count=3, logprobs=[-0.5, -1.2, -0.8, -0.3, -1.5, -2.0])
        ['789', '123', '456']  # Sorted by logprobs first
        >>> get_unique_generations(gens, max_count=3, exclude_sids={'456', '789'})
        ['123', '999']  # Excluded '456' and '789'
        >>> get_unique_generations(gens, max_count=3, sources=['a', 'b', 'a', 'c', 'b', 'd'])
        (['123', '456', '789'], ['a', 'b', 'c'])
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
    unique_sids = []
    unique_sources = [] if track_sources else None
    exclude = exclude_sids or set()

    for i, gen in enumerate(sorted_generations):
        # Skip empty strings
        if not gen or not gen.strip():
            continue

        # Extract SID from generation text
        sid = extract_id_from_generation(gen)

        # Skip if SID is empty, already seen, or in exclude list
        if not sid or sid in seen or sid in exclude:
            continue

        unique_sids.append(sid)
        seen.add(sid)

        if track_sources:
            unique_sources.append(sorted_sources[i])

        # Stop if we've collected enough unique SIDs
        if len(unique_sids) >= max_count:
            break

    if track_sources:
        return unique_sids, unique_sources
    return unique_sids


def get_debug_info(
    sample_id: str,
    generations: List[str],
    ground_truth: str,
    pass_results: Dict[str, bool],
    position1_pass_results: Dict[str, bool],
    raw_prompt: str = ""
) -> Dict[str, Any]:
    """
    Prepare debug information for a sample

    Args:
        sample_id: Sample ID
        generations: List of generated SIDs
        ground_truth: Ground truth answer string
        pass_results: Pass@k results for this sample
        position1_pass_results: Position1_Pass@k results for this sample
        raw_prompt: Raw prompt (optional)

    Returns:
        Debug information dictionary
    """
    ground_truth_sids = extract_ids_from_answer(ground_truth)
    first_ground_truth_sid = extract_first_id_from_answer(ground_truth)

    # Extract top-k generated IDs
    top_k_sids = [extract_id_from_generation(gen) for gen in generations[:10]]  # Show top-10

    debug_item = {
        "sample_id": sample_id,
        "ground_truth_sids": list(ground_truth_sids),
        "first_ground_truth_sid": first_ground_truth_sid,
        "top_10_generations": top_k_sids,
        "pass_results": pass_results,
        "position1_pass_results": position1_pass_results,
    }

    if raw_prompt:
        debug_item["raw_prompt_snippet"] = raw_prompt[:200] + "..." if len(raw_prompt) > 200 else raw_prompt

    return debug_item
