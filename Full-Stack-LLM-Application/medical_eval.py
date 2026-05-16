from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Set
import re

# Use the same embedding model you already load, or a lightweight dedicated one
_eval_model = None

def _get_eval_model():
    global _eval_model
    if _eval_model is None:
        # Lightweight medical-sentence model; swap for your own if preferred
        _eval_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli", device="cpu")
    return _eval_model


def calculate_semantic_similarity(predictions: List[str], references: List[str]) -> Dict:
    """
    BERTScore-style cosine similarity using your existing sentence-transformers stack.
    Captures medical paraphrasing (e.g., 'MI' vs 'myocardial infarction').
    """
    model = _get_eval_model()
    pred_emb = model.encode(predictions, convert_to_tensor=True, show_progress_bar=False)
    ref_emb = model.encode(references, convert_to_tensor=True, show_progress_bar=False)
    cosine_scores = np.diag(np.inner(pred_emb, ref_emb) / (
        np.linalg.norm(pred_emb, axis=1) * np.linalg.norm(ref_emb, axis=1) + 1e-8
    ))
    return {
        "semantic_similarity_mean": round(float(cosine_scores.mean()), 4),
        "semantic_similarity_scores": [round(float(s), 4) for s in cosine_scores],
    }


def extract_medical_entities(text: str) -> Set[str]:
    """
    Lightweight medical entity extractor.
    In production, replace with scispaCy or MedSpaCy. Here: simple keyword + regex.
    """
    # Lowercase and tokenize roughly
    t = text.lower()
    # Common medical patterns: drugs, doses, anatomical terms, diagnoses
    # This is a starter regex set — expand with your domain terms
    patterns = [
        r'\b\d+\s*(?:mg|mcg|g|ml|iu|units?)\b',           # doses
        r'\b(?:aspirin|metformin|lisinopril|atorvastatin|amoxicillin|prednisone)\b',
        r'\b(?:myocardial infarction|heart attack|stroke|pneumonia|diabetes|hypertension)\b',
        r'\b(?:chest x-ray|ct scan|mri|ultrasound|ecg|eeg)\b',
        r'\b(?:patient|diagnosis|prognosis|symptom|syndrome)\b',
    ]
    entities = set()
    for pat in patterns:
        entities.update(re.findall(pat, t))
    return entities


def calculate_entity_f1(predictions: List[str], references: List[str]) -> List[Dict]:
    """
    Token-set F1 over extracted medical entities.
    Good for: did the answer mention the correct drugs, diagnoses, anatomy?
    """
    results = []
    for pred, ref in zip(predictions, references):
        p_set = extract_medical_entities(pred)
        r_set = extract_medical_entities(ref)
        overlap = len(p_set & r_set)
        precision = overlap / len(p_set) if p_set else 0.0
        recall = overlap / len(r_set) if r_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append({
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "pred_entities": list(p_set),
            "ref_entities": list(r_set),
        })
    return results


def calculate_retrieval_faithfulness(answer: str, retrieved_context: str) -> Dict:
    """
    Did the LLM hallucinate facts not in the retrieved PubMed/KB context?
    Simple implementation: entity coverage + semantic similarity to context.
    """
    sem = calculate_semantic_similarity([answer], [retrieved_context])
    ans_ents = extract_medical_entities(answer)
    ctx_ents = extract_medical_entities(retrieved_context)
    covered = len(ans_ents & ctx_ents) / len(ans_ents) if ans_ents else 1.0
    return {
        "context_entity_coverage": round(covered, 4),
        "answer_context_similarity": sem["semantic_similarity_mean"],
        "hallucination_flag": covered < 0.5 and sem["semantic_similarity_mean"] < 0.6,
    }