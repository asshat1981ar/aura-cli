"""
Evaluation harness for ASCM retrieval quality.
Calculates Recall@k (k=1, 3, 5) and Mean Reciprocal Rank (MRR).
"""
import json
import numpy as np
import sqlite3
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unittest.mock import MagicMock
from core.vector_store import VectorStore
from core.memory_types import MemoryRecord, RetrievalQuery

# Re-use deterministic vectors from retrieval quality test
VECTORS = {
    "rec_1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "rec_2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "rec_3": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "rec_4": np.array([0.0, 0.9, 0.1, 0.0], dtype=np.float32),
    "q_auth": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "q_db":   np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "q_math": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
}

class EvalAdapter:
    def __init__(self):
        self._model = "eval-model"
    def model_id(self): return self._model
    def dimensions(self): return 4
    def embed(self, texts):
        results = []
        for t in texts:
            if "calculate_metric" in t or "calculate metric" in t: results.append(VECTORS["rec_1"])
            elif "UserAuth" in t or "authenticate" in t: results.append(VECTORS["q_auth"])
            elif "save_to_db" in t or "database" in t: results.append(VECTORS["q_db"])
            elif "middleware" in t: results.append(VECTORS["rec_4"])
            else: results.append(np.zeros(4, dtype=np.float32))
        return results
    def get_embedding(self, text): return self.embed([text])[0]

def run_eval():
    print("ASCM Retrieval Quality Evaluation")
    print("=" * 80)
    
    # Setup
    db_path = ":memory:"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    brain = MagicMock()
    brain.db = conn
    vs = VectorStore(EvalAdapter(), brain)
    
    golden_path = Path(__file__).parent / "fixtures" / "retrieval_golden_set.json"
    with open(golden_path, "r") as f:
        golden_set = json.load(f)
        
    # Index
    records = []
    for r in golden_set["records"]:
        vec = VECTORS.get(r["id"])
        records.append(MemoryRecord(
            id=r["id"], 
            content=r["content"], 
            source_type="eval", 
            source_ref=r["source_ref"],
            created_at=time.time(), 
            updated_at=time.time(), 
            content_hash=r["id"],
            embedding=vec.tobytes() if vec is not None else None,
            embedding_model="eval-model",
            embedding_dims=4
        ))
    vs.upsert(records)
    
    # Metrics
    k_list = [1, 3, 5]
    total_mrr = 0.0
    queries = golden_set["queries"]
    
    header = f"{'Query Text':<40} | {'Rec@1':<6} | {'Rec@3':<6} | {'Rec@5':<6} | {'MRR':<6}"
    print(header)
    print("-" * len(header))
    
    summary_recalls = {k: 0.0 for k in k_list}
    
    for q_case in queries:
        expected = set(q_case["expected_ids"])
        hits = vs.search(RetrievalQuery(query_text=q_case["text"], k=5, min_score=0.1))
        hit_ids = [h.record_id for h in hits]
        
        # Recall@k
        recalls = {}
        for k in k_list:
            found = set(hit_ids[:k]) & expected
            recall_val = len(found) / len(expected) if expected else 0.0
            recalls[k] = recall_val
            summary_recalls[k] += recall_val
            
        # MRR
        rr = 0.0
        for i, hid in enumerate(hit_ids, 1):
            if hid in expected:
                rr = 1.0 / i
                break
        total_mrr += rr
        
        print(f"{q_case['text'][:40]:<40} | {recalls[1]:.2f}  | {recalls[3]:.2f}  | {recalls[5]:.2f}  | {rr:.2f}")
        
    num_queries = len(queries)
    avg_mrr = total_mrr / num_queries if num_queries else 0.0
    print("-" * len(header))
    print(f"{'AVERAGE':<40} | {summary_recalls[1]/num_queries:.2f}  | {summary_recalls[3]/num_queries:.2f}  | {summary_recalls[5]/num_queries:.2f}  | {avg_mrr:.2f}")
    print("=" * 80)
    print(f"Final Score (MRR): {avg_mrr:.3f}")

if __name__ == "__main__":
    run_eval()
