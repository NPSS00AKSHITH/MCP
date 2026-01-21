
import logging
import json
import os
import sys
from unittest.mock import MagicMock
from src.server.loop import RetrievalLoop
from src.evaluation.evaluator import LogProcessor, Evaluator, ThresholdTuner
from src.server.logging import bind_logger_context

# Configure logging to file for parsing
LOG_FILE = "eval_test.log"
import structlog

def run_evaluation_pipeline():
    # 1. Generate Data
    print("Generating traces...")
    # Clean old log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    # Redirect structlog to file
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars, # Required to pick up bound context
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.WriteLoggerFactory(file=open(LOG_FILE, "w")),
    )
    
    # Run traces (simulating real usage)
    from examples.generate_traces import run_trace, setup_skipped, setup_single_pass, setup_multi_step, setup_ambiguous, setup_failure
    
    # Wrap to bind request_id
    def run_with_id(name, setup, req_id):
        bind_logger_context(request_id=req_id)
        run_trace(name, setup)
        
    run_with_id("Skipped", setup_skipped, "req_1")
    run_with_id("Single", setup_single_pass, "req_2")
    run_with_id("Multi", setup_multi_step, "req_3")
    run_with_id("Ambiguous", setup_ambiguous, "req_4")
    run_with_id("Failed", setup_failure, "req_5")
    
    # 2. Process Logs
    print("Processing logs...")
    processor = LogProcessor()
    df = processor.process(LOG_FILE)
    print(f"Loaded {len(df)} records.")
    
    # 3. Compute Metrics
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(df)
    
    # 4. Tune Thresholds
    tuner = ThresholdTuner()
    new_thresh = {"top_score": 0.8, "gap": 0.2, "entropy": 0.5} # Stricter
    sim_df = tuner.simulate(df, new_thresh)

    # 5. Generate Report
    report_content = f"""# Adaptive RAG Evaluation Report

## Summary Metrics
```json
{json.dumps(metrics, indent=2)}
```

## Trace Analysis
Total Queries: {len(df)}

{df.to_markdown()}

## Threshold Simulation (Stricter)
New Thresholds: {new_thresh}

{sim_df.to_markdown()}
"""
    with open("evaluation_report.md", "w") as f:
        f.write(report_content)
    
    print("\nReport saved to evaluation_report.md")


if __name__ == "__main__":
    run_evaluation_pipeline()
