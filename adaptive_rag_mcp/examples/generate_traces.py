
import logging
from unittest.mock import MagicMock
from src.server.loop import RetrievalLoop
from src.server.policy import QueryType, RetrievalDecision
from src.retrieval.reranker import QualitySignals

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_trace(name, scenario_setup):
    print(f"\n--- TRACE: {name} ---")
    
    # Mock components
    loop = RetrievalLoop()
    loop.policy = MagicMock()
    loop.retriever = MagicMock()
    loop.reranker = MagicMock()
    
    # Setup scenario
    scenario_setup(loop)
    
    # Run
    result = loop.run("Test query")
    
    # Print Trace
    print(f"Goal: {name}")
    print(f"Success: {result.success}")
    print(f"Reason: {result.reason}")
    print("Steps:")
    for step in result.steps:
        print(f"  Step {step.step_number}: Strategy={step.strategy}, Retrieved={step.retrieved_count}, TopScore={step.top_score}, Confident={step.is_confident}")
    if not result.steps and result.success:
        print("  (No steps - skipped)")

from src.server.logging import get_logger

logger = get_logger("trace_gen")

def setup_skipped(loop):
    decision = RetrievalDecision(should_retrieve=False, query_type=QueryType.GENERAL_KNOWLEDGE, reason="General knowledge")
    loop.policy.decide.return_value = decision
    # Simulate the log that PolicyEngine would have produced
    logger.info("policy_decision", query_type=decision.query_type.value, should_retrieve=decision.should_retrieve, mode=decision.search_mode, k=decision.max_k, reason=decision.reason)

def setup_single_pass(loop):
    decision = RetrievalDecision(should_retrieve=True, query_type=QueryType.DOC_SPECIFIC, reason="Specific doc", search_mode="hybrid", max_k=5)
    loop.policy.decide.return_value = decision
    logger.info("policy_decision", query_type=decision.query_type.value, should_retrieve=decision.should_retrieve, mode=decision.search_mode, k=decision.max_k, reason=decision.reason)
    
    # High score signals
    signals = QualitySignals(top_score=0.95, score_gap=0.5, evidence_entropy=0.2, score_spread=0.2, mean_score=0.5, relevant_count=1, total_count=5, confidence_flags=[])
    
    loop.retriever.search.return_value = [MagicMock(chunk_id="1", content="content", metadata={})]
    loop.reranker.rerank.return_value = ([MagicMock(relevance_score=0.95)], signals)
    loop.policy.evaluate_evidence.return_value = True

def setup_multi_step(loop):
    decision = RetrievalDecision(should_retrieve=True, query_type=QueryType.DOC_SPECIFIC, reason="Hard query", search_mode="hybrid", max_k=5)
    loop.policy.decide.return_value = decision
    logger.info("policy_decision", query_type=decision.query_type.value, should_retrieve=decision.should_retrieve, mode=decision.search_mode, k=decision.max_k, reason=decision.reason)

    # Step 1: Low score
    signals_low = QualitySignals(top_score=0.4, score_gap=0.0, evidence_entropy=0.9, score_spread=0.1, mean_score=0.3, relevant_count=0, total_count=5, confidence_flags=["low_top_score"])
    
    # Step 2: High score
    signals_high = QualitySignals(top_score=0.88, score_gap=0.3, evidence_entropy=0.4, score_spread=0.2, mean_score=0.6, relevant_count=2, total_count=10, confidence_flags=[])

    # Mock sequence
    loop.retriever.search.side_effect = [
        [MagicMock(chunk_id="1", content="bad", metadata={})], # Step 1
        [MagicMock(chunk_id="2", content="good", metadata={})] # Step 2
    ]
    loop.reranker.rerank.side_effect = [
        ([MagicMock(relevance_score=0.4)], signals_low),
        ([MagicMock(relevance_score=0.88)], signals_high)
    ]
    loop.policy.evaluate_evidence.side_effect = [False, True]
    
    # Retry params
    loop.policy.determine_retry_strategy.return_value = {"search_mode": "dense", "max_k": 10, "query": "Test query expanded"}

def setup_ambiguous(loop):
    decision = RetrievalDecision(should_retrieve=False, query_type=QueryType.AMBIGUOUS, reason="Ambiguous query")
    loop.policy.decide.return_value = decision
    logger.info("policy_decision", query_type=decision.query_type.value, should_retrieve=decision.should_retrieve, mode=decision.search_mode, k=decision.max_k, reason=decision.reason)

def setup_failure(loop):
    decision = RetrievalDecision(should_retrieve=True, query_type=QueryType.DOC_SPECIFIC, reason="Hard unknown query", search_mode="hybrid", max_k=5)
    loop.policy.decide.return_value = decision
    logger.info("policy_decision", query_type=decision.query_type.value, should_retrieve=decision.should_retrieve, mode=decision.search_mode, k=decision.max_k, reason=decision.reason)
    
    # Always low score
    signals_low = QualitySignals(top_score=0.3, score_gap=0.0, evidence_entropy=0.8, score_spread=0.1, mean_score=0.2, relevant_count=0, total_count=5, confidence_flags=["low_top_score"])
    
    loop.retriever.search.return_value = [MagicMock(chunk_id="1", content="irrelevant", metadata={})]
    loop.reranker.rerank.return_value = ([MagicMock(relevance_score=0.3)], signals_low)
    
    loop.policy.evaluate_evidence.return_value = False
    loop.policy.determine_retry_strategy.return_value = None # Stop retrying

if __name__ == "__main__":
    run_trace("Skipped Retrieval", setup_skipped)
    run_trace("Single Pass Retrieval", setup_single_pass)
    run_trace("Multi-Step Adaptive Retrieval", setup_multi_step)
    run_trace("Ambiguous Query", setup_ambiguous)
    run_trace("Failed Retrieval", setup_failure)
