# Example Execution Traces for Epistemic Safety

This document provides example execution traces demonstrating each outcome type
and how the epistemic safety features work in practice.

---

## Trace 1: answer_ready (High Confidence)

```
Query: "What is the budget allocation for Q3?"

[Policy Decision]
  intent: factual_lookup
  retrieval_mode: hybrid
  should_retrieve: true

[Iteration 1]
  mode: hybrid
  k: 5
  chunks_retrieved: 5
  chunks_reranked: 5
  top_score: 0.85
  score_gap: 0.22
  entropy: 0.45
  
[Contradiction Check]
  has_contradiction: false
  
[Stopping Rule Evaluation]
  rule_triggered: HIGH_CONFIDENCE
  stopped: true
  reason: "top_score >= 0.7 AND score_gap >= 0.15"

[Final Outcome]
  outcome_type: answer_ready
  confidence_level: high
  explanation: "High confidence evidence found. Top score: 0.85, score gap: 0.22. No contradictions detected."

[Structured Log]
{
  "event": "outcome_selected",
  "outcome_type": "answer_ready",
  "confidence_level": "high",
  "explanation": "High confidence evidence found...",
  "why": "System chose this outcome based on evidence scores and stopping rules."
}
```

---

## Trace 2: partial_answer (Moderate Confidence)

```
Query: "Summarize the project milestones"

[Policy Decision]
  intent: summarization
  retrieval_mode: hybrid
  should_retrieve: true

[Iteration 1]
  mode: hybrid
  k: 5
  chunks_retrieved: 5
  chunks_reranked: 5
  top_score: 0.55
  score_gap: 0.12
  entropy: 0.65
  is_confident: false
  
[Iteration 2 - Retry]
  mode: dense
  k: 8
  chunks_retrieved: 8
  chunks_reranked: 8
  top_score: 0.58
  is_confident: false
  
[HyDE Fallback Triggered]
  reason: "Low confidence evidence, trying HyDE"
  hypothetical_generated: true
  
[Iteration 2 - HyDE]
  mode: dense_hyde
  k: 8
  top_score: 0.62
  is_confident: false
  improvement: true (0.58 -> 0.62)

[Iteration 3 - Final]
  mode: hybrid
  k: 12
  top_score: 0.64
  is_confident: false

[Stopping Rule Evaluation]
  rule_triggered: MAX_ITERATIONS
  stopped: true

[Final Outcome]
  outcome_type: partial_answer
  confidence_level: medium
  explanation: "Some evidence found but confidence is medium (top_score=0.64)."

[Structured Log]
{
  "event": "outcome_selected",
  "outcome_type": "partial_answer",
  "confidence_level": "medium",
  "explanation": "Some evidence found but confidence is medium..."
}
```

---

## Trace 3: insufficient_evidence (Low Confidence)

```
Query: "What is the CEO's favorite color?"

[Policy Decision]
  intent: factual_lookup
  retrieval_mode: hybrid
  should_retrieve: true

[Iteration 1]
  mode: hybrid
  k: 5
  chunks_retrieved: 5
  chunks_reranked: 5
  top_score: 0.25
  score_gap: 0.08
  entropy: 0.82
  is_confident: false

[Iteration 2]
  mode: dense
  k: 8
  top_score: 0.28
  is_confident: false

[HyDE Fallback Triggered]
  hypothetical_generated: true
  top_score after HyDE: 0.30
  improvement: marginal

[Iteration 3]
  mode: hybrid
  k: 12
  top_score: 0.28
  is_confident: false

[Stopping Rule Evaluation]
  rule_triggered: MAX_ITERATIONS
  stopped: true
  confidence_level: low (top_score < 0.5)

[Final Outcome]
  outcome_type: insufficient_evidence
  confidence_level: low
  explanation: "Max iterations reached with low confidence (top_score=0.28)."

[Structured Log]
{
  "event": "outcome_selected",
  "outcome_type": "insufficient_evidence",
  "confidence_level": "low",
  "why": "System chose this outcome because evidence quality was too low."
}
```

---

## Trace 4: clarification_needed (Ambiguous Query)

```
Query: "the bank"

[Policy Decision]
  intent: ambiguous
  retrieval_mode: none
  should_retrieve: false
  decision_reason: "Query is ambiguous - clarification may be needed."

[Stopping Rule Evaluation]
  rule_triggered: POLICY_SKIP
  stopped: true

[Final Outcome]
  outcome_type: clarification_needed
  confidence_level: low
  explanation: "Query is ambiguous - clarification may be needed."

[Structured Log]
{
  "event": "outcome_selected",
  "outcome_type": "clarification_needed",
  "confidence_level": "low",
  "why": "Policy engine detected ambiguous query."
}
```

---

## Trace 5: partial_answer with Contradiction Detection

```
Query: "What is the project deadline?"

[Policy Decision]
  intent: factual_lookup
  retrieval_mode: hybrid
  should_retrieve: true

[Iteration 1]
  mode: hybrid
  k: 5
  chunks_retrieved: 5
  chunks_reranked: 5
  top_score: 0.82
  score_gap: 0.18

[Contradiction Check]
  has_contradiction: true
  conflicting_chunks: [["chunk_001", "chunk_007"]]
  
  chunk_001: "The project deadline is December 15th, 2024."
  chunk_007: "Project deadline has been extended to January 30th, 2025."
  
  similarity: 0.45 (related topic)
  reason: "Different dates detected for same topic"
  
  explanation: "Potential contradictions detected between chunks: chunk_001 and chunk_007. High-confidence sources appear to disagree. Unable to merge claims safely."

[Stopping Rule Evaluation]
  rule_triggered: CONTRADICTION_DETECTED
  stopped: true

[Final Outcome]
  outcome_type: partial_answer
  confidence_level: low
  explanation: "Conflicting sources detected (top_score=0.82). Potential contradictions detected between chunks: chunk_001 and chunk_007. High-confidence sources appear to disagree. Unable to merge claims safely."

[Structured Log]
{
  "event": "contradiction_detected",
  "iteration": 1,
  "conflicting_chunks": [["chunk_001", "chunk_007"]],
  "explanation": "Potential contradictions detected..."
}

{
  "event": "outcome_selected",
  "outcome_type": "partial_answer",
  "confidence_level": "low",
  "why": "Despite high top_score, contradictions prevent answer_ready."
}
```

---

## Key Observations

1. **Explicit Outcomes**: Every retrieval ends with a clear outcome type and explanation.

2. **Confidence Levels**: Confidence is explicitly calculated and returned.

3. **No Silent Stopping**: Every stop has a logged reason and triggered rule.

4. **Contradiction Safety**: When sources conflict, the system:
   - Does NOT average or merge conflicting claims
   - Returns `partial_answer` instead of `answer_ready`
   - Sets `confidence_level` to `low`
   - Explains which chunks conflict

5. **Prefer Refusing Over Guessing**: If confidence is low, the system admits uncertainty rather than fabricating answers.

6. **Logging & Explainability**: Structured logs answer "Why did the system choose this outcome?"
