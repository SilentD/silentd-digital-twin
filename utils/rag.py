import json
import numpy as np
import spacy

# Load a lightweight local embedder (Make sure to run: python -m spacy download en_core_web_md)
_nlp = spacy.load(r".venv/lib/python3.12/site-packages/en_core_web_md/en_core_web_md-3.8.0/")
print("done")

def process_json_knowledge(json_path: str) -> dict:
    """Reads a JSON file, flattens its elements into structured text chunks, and vectorises them."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    chunks = []
    
    # 1. Extract named profile facts
    if "digital_twin_profile" in data:
        profile = data["digital_twin_profile"]
        profile_str = (
            f"Profile: Name is {profile.get('name')}. "
            f"Specialized fields: {profile.get('specialty')}. "
            f"Core Tech Stack: {', '.join(profile.get('tech_stack', []))}."
        )
        chunks.append(profile_str)
        
    # 2. Extract standard background facts list
    if "facts" in data:
        for fact in data["facts"]:
            if fact.strip():
                chunks.append(fact.strip())
                
    if not chunks:
        return {"chunks": [], "embeddings": np.array([])}

    # Generate matrix vectors using spaCy pipeline processing
    # .vector extracts the averaged dense embedding for the entire sentence block
    embeddings = np.array([_nlp(chunk).vector for chunk in chunks])
    
    # Normalize the vectors to unit length so dot product exactly equals cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Prevent division by zero if a chunk happens to be completely empty of vectors
    norms[norms == 0] = 1.0 
    normalized_embeddings = embeddings / norms
    
    return {"chunks": chunks, "embeddings": normalized_embeddings}

def retrieve_context(query: str, vector_db: dict, top_k: int = 2) -> str:
    """Finds the most relevant textual facts matching the query via Cosine Similarity."""
    if not vector_db or len(vector_db.get("chunks", [])) == 0:
        return ""
    
    # Extract query vector and normalize it
    query_vector = _nlp(query).vector
    query_norm = np.linalg.norm(query_vector)
    if query_norm > 0:
        query_vector = query_vector / query_norm
    
    # Compute Cosine Similarity matching scores via matrix multiplication
    # Because both vectors are normalized, np.dot is mathematically identical to Cosine Similarity
    scores = np.dot(vector_db["embeddings"], query_vector)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    retrieved_chunks = [vector_db["chunks"][idx] for idx in top_indices]
    return "\n\n".join(retrieved_chunks)
