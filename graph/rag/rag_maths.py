import numpy as np

def compute_top_k(query_vector: list, registry_records: list, db_embeddings: np.ndarray, k: int, lambda_mult: float = 0.5) -> list:
    """Executes a highly optimized SIMD Vectorized MMR selection pass without list allocation overhead."""
    if not query_vector or registry_records is None or db_embeddings is None: return []

    q_arr = np.array(query_vector, dtype=np.float32)
    dot_products = np.dot(db_embeddings, q_arr)
    db_norms = np.linalg.norm(db_embeddings, axis=1)
    q_norm = np.linalg.norm(q_arr)
    
    denominators = db_norms * q_norm
    denominators[denominators == 0.0] = 1.0
    similarities = dot_products / denominators

    if len(registry_records) <= k:
        sorted_indices = np.argsort(similarities)[::-1]
        return [_entry(registry_records[idx], similarities[idx]) for idx in sorted_indices]

    selected_indices = []
    unselected_indices = list(range(len(registry_records)))
    
    first_choice = int(np.argmax(similarities))
    selected_indices.append(first_choice)
    unselected_indices.remove(first_choice)

    while len(selected_indices) < k:
        remaining_sims = similarities[unselected_indices]
        remaining_embeddings = db_embeddings[unselected_indices]
        selected_embeddings = db_embeddings[selected_indices]
        
        cross_dot = np.dot(remaining_embeddings, selected_embeddings.T)
        rem_norms = db_norms[unselected_indices, np.newaxis]
        sel_norms = db_norms[selected_indices, np.newaxis].T
        
        cross_denominators = rem_norms * sel_norms
        cross_denominators[cross_denominators == 0.0] = 1.0
        cross_sims = cross_dot / cross_denominators
        
        max_cross_sim = np.max(cross_sims, axis=1)
        mmr_scores = lambda_mult * remaining_sims - (1.0 - lambda_mult) * max_cross_sim
        
        winner_idx_in_remaining = np.argmax(mmr_scores)
        actual_winner_idx = unselected_indices[winner_idx_in_remaining]
        
        selected_indices.append(actual_winner_idx)
        unselected_indices.remove(actual_winner_idx)

    return [_entry(registry_records[idx], similarities[idx]) for idx in selected_indices]


def _entry(record: dict, score) -> dict:
    """Build a retrieval-result dict from a registry record + similarity score.

    Excludes the bulky `embedding` vector — callers don't need it.
    """
    return {
        "score": float(score),
        "content": record.get("content", ""),
        "source": record.get("source", "unknown"),
        "scope": record.get("scope", "__GLOBAL__"),
    }
