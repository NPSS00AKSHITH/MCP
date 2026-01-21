
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.retrieval.reranker import QualitySignals
from src.server.policy import PolicyEngine, RetrievalDecision

@dataclass
class EvalRecord:
    request_id: str
    query: str
    intent: str
    final_intent_conf: float # Not tracked yet, but good to have schema
    retrieval_strategy: str
    iterations: int
    total_context_size: int
    top_score: float
    score_gap: float
    entropy: float
    skipped: bool
    final_outcome: str # "success", "max_iterations", "skipped"
    latency: float
    
class LogProcessor:
    """Parses JSON logs into structured evaluation dataset."""
    
    def process(self, log_file_path: str) -> pd.DataFrame:
        records = {}
        
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    req_id = entry.get("request_id", "unknown")
                    
                    if req_id not in records:
                        records[req_id] = {
                            "request_id": req_id,
                            "query": "",
                            "intent": "unknown",
                            "iterations": 0,
                            "total_context_size": 0,
                            "top_score": 0.0,
                            "score_gap": 0.0,
                            "entropy": 0.0,
                            "skipped": False,
                            "final_outcome": "unknown",
                            "latency": 0.0
                        }
                    
                    rec = records[req_id]
                    event = entry.get("event")
                    
                    if event == "tool_call_started":
                        rec["query"] = entry.get("input_keys", []) # input_data dict parsing needed usually
                        # But log_tool_call logs keys, not values... 
                        # Ah, the current logging implementation only logs KEYS.
                        # This is a problem! We need the QUERY to be logged.
                        # Wait, loop logs "loop_iteration_start" with query!
                        pass
                        
                    if event == "loop_iteration_start":
                        rec["query"] = entry.get("query", rec["query"])
                        
                    if event == "policy_decision":
                        rec["intent"] = entry.get("query_type")
                        rec["retrieval_strategy"] = entry.get("mode")
                        if not entry.get("should_retrieve"):
                            rec["skipped"] = True
                            rec["final_outcome"] = "skipped"
                            
                    if event == "loop_iteration_eval":
                        rec["iterations"] = entry.get("iteration")
                        rec["total_context_size"] += entry.get("context_size", 0)
                        # Keep better score if multi-step? Or last step?
                        # Usually we care about the *final* retrieved result.
                        rec["top_score"] = entry.get("top_score")
                        
                    if event == "reranking_complete":
                         rec["score_gap"] = entry.get("score_gap", 0)
                         rec["entropy"] = entry.get("entropy", 0)
                         
                    if event == "tool_call_success":
                        rec["latency"] = entry.get("duration_ms", 0)
                        if not rec["skipped"]:
                             # Infer success/fail
                             # We'd need to parse the RESULT payload, but we don't log it fully
                             # We can infer from loop_iteration_eval "confident" flag if last one
                             pass

                except json.JSONDecodeError:
                    continue
                    
        return pd.DataFrame.from_dict(records, orient='index')

class Evaluator:
    """Computes metrics and failure modes."""
    
    def compute_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
            
        return {
            "retrieval_skip_rate": df["skipped"].mean(),
            "avg_iterations": df[~df["skipped"]]["iterations"].mean(),
            "avg_context_size": df["total_context_size"].mean(),
            "avg_latency": df["latency"].mean(),
            "success_rate": 0.0 # Placeholder
        }
    
    def detect_failures(self, row) -> List[str]:
        failures = []
        if row["skipped"] and row["intent"] == "doc_specific":
            failures.append("wrong_retriever_choice") # Should not skip doc specific
            
        if row["iterations"] >= 3 and row["top_score"] < 0.6:
            failures.append("under_retrieval")
            
        return failures

class ThresholdTuner:
    """Simulates outcomes with different thresholds."""
    
    def __init__(self):
        self.policy = PolicyEngine() # Use real policy logic
        
    def simulate(self, df: pd.DataFrame, new_thresholds: Dict[str, float]) -> pd.DataFrame:
        # Clone policy to mess with it
        # Actually easier to just call evaluate_evidence with a dummy object
        
        simulated_results = []
        
        # Patch constants locally or use a helper method that takes args?
        # PolicyEngine methods use module-level constants. Ideally we'd refactor PolicyEngine to 
        # accept config. For offline sim, let's just create a dummy "signals" object 
        # and checking the logic manually or refactoring policy.
        # For now, I will REIMPLEMENT the check logic here to allow flexible thresholds
        # This duplicates logic but satisfies "offline analysis" without changing prod code too much.
        
        high_conf = new_thresholds.get("top_score", 0.7)
        gap_thresh = new_thresholds.get("gap", 0.15)
        entropy_thresh = new_thresholds.get("entropy", 0.6)
        
        for _, row in df.iterrows():
            # Simulate "is_confident"
            is_confident = False
            if row["top_score"] >= high_conf:
                if row["score_gap"] >= gap_thresh or row["top_score"] > 0.85:
                    if row["entropy"] <= entropy_thresh:
                        is_confident = True
            
            simulated_results.append({
                "request_id": row["request_id"],
                "original_outcome": "success" if row["top_score"] > 0.7 else "fail", # Approximate
                "simulated_outcome": "success" if is_confident else "fail"
            })
            
        return pd.DataFrame(simulated_results)
