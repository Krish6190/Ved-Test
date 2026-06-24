import numpy as np

def compute_top_k(query_vector: list, registry_records: list, k: int, lambda_mult: float = 0.5) -> list:
    """
    Executes a high-speed Vectorized Maximal Marginal Relevance (MMR) selection pass.
    Allows dynamic lambda_mult adjustments (0.0 = maximum diversity, 1.0 = pure similarity).
    """
    if not query_vector or not registry_records:
        return []

    # 1. Instantiate fast contiguous memory C-arrays for SIMD calculations
    q_arr = np.array(query_vector, dtype=np.float32)
    db_embeddings = np.array([r["embedding"] for r in registry_records], dtype=np.float32)
    
    # 2. Pre-calculate standard similarities across the database
    dot_products = np.dot(db_embeddings, q_arr)
    db_norms = np.linalg.norm(db_embeddings, axis=1)
    q_norm = np.linalg.norm(q_arr)
    
    denominators = db_norms * q_norm
    denominators[denominators == 0.0] = 1.0
    similarities = dot_products / denominators

    # If the database has fewer entries than requested, return standard similarities immediately
    if len(registry_records) <= k:
        sorted_indices = np.argsort(similarities)[::-1]
        return [{"score": float(similarities[idx]), "content": registry_records[idx]["content"]} for idx in sorted_indices]

    # 3. Initialize the MMR iterative tracking selection arrays
    selected_indices = []
    unselected_indices = list(range(len(registry_records)))
    
    # The first document chosen is always the absolute highest scoring similarity chunk
    first_choice = int(np.argmax(similarities))
    selected_indices.append(first_choice)
    unselected_indices.remove(first_choice)

    # 4. Run the high-speed MMR diversity ranking loop
    while len(selected_indices) < k:
        remaining_sims = similarities[unselected_indices]
        remaining_embeddings = db_embeddings[unselected_indices]
        selected_embeddings = db_embeddings[selected_indices]
        
        # Calculate cross-similarity between remaining unselected chunks and already selected chunks
        # Formula: Matrix multiplication of remaining items vs selected items
        cross_dot = np.dot(remaining_embeddings, selected_embeddings.T)
        
        # Norm arrays for normalizations
        rem_norms = db_norms[unselected_indices, np.newaxis]
        sel_norms = db_norms[selected_indices, np.newaxis].T
        cross_denominators = rem_norms * sel_norms
        cross_denominators[cross_denominators == 0.0] = 1.0
        
        # Similarity matrix of remaining docs vs selected docs
        cross_sims = cross_dot / cross_denominators
        
        # Find the maximum similarity score to ANY selected doc for each remaining chunk
        max_cross_sim = np.max(cross_sims, axis=1)
        
        # MMR Equation Core: Equation balances relevance (term 1) against redundancy (term 2)
        mmr_scores = lambda_mult * remaining_sims - (1.0 - lambda_mult) * max_cross_sim
        
        # Pick the winner index for this diversity iteration pass
        winner_idx_in_remaining = np.argmax(mmr_scores)
        actual_winner_idx = unselected_indices[winner_idx_in_remaining]
        
        selected_indices.append(actual_winner_idx)
        unselected_indices.remove(actual_winner_idx)

    return [
        {"score": float(similarities[idx]), "content": registry_records[idx]["content"]}
        for idx in selected_indices
    ]
