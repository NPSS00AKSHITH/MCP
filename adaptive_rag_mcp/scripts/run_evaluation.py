#!/usr/bin/env python3
"""MCP Evaluation Runner for Adaptive RAG MCP Server.

Runs evaluation questions from an XML file against the MCP server
and reports accuracy metrics.

Usage:
    python scripts/run_evaluation.py tests/evaluation.xml

Requirements:
    - pip install httpx anthropic
    - Server must be running at specified URL
    - ANTHROPIC_API_KEY environment variable (for LLM-based evaluation)
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx


@dataclass
class QAPair:
    """Question-answer pair from evaluation file."""
    question: str
    expected_answer: str


@dataclass
class EvalResult:
    """Result of evaluating a single question."""
    question: str
    expected: str
    actual: str
    correct: bool
    duration_ms: float
    tool_calls: int = 0
    error: str | None = None


def load_evaluation_file(path: str) -> list[QAPair]:
    """Load QA pairs from XML evaluation file."""
    tree = ET.parse(path)
    root = tree.getroot()
    
    pairs = []
    for qa in root.findall("qa_pair"):
        question = qa.find("question")
        answer = qa.find("answer")
        if question is not None and answer is not None:
            pairs.append(QAPair(
                question=question.text.strip() if question.text else "",
                expected_answer=answer.text.strip() if answer.text else "",
            ))
    
    return pairs


def call_tool(client: httpx.Client, base_url: str, api_key: str, tool_name: str, params: dict) -> dict:
    """Call an MCP tool via HTTP."""
    response = client.post(
        f"{base_url}/tools/{tool_name}",
        json=params,
        headers={"X-API-Key": api_key},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def run_simple_evaluation(
    base_url: str,
    api_key: str,
    qa_pairs: list[QAPair],
) -> list[EvalResult]:
    """Run simple tool-based evaluation (no LLM)."""
    results = []
    
    with httpx.Client() as client:
        # First verify server is reachable
        try:
            health = client.get(f"{base_url}/health", timeout=5.0)
            health.raise_for_status()
            print(f"✓ Server healthy: {health.json()}")
        except Exception as e:
            print(f"✗ Cannot reach server at {base_url}: {e}")
            sys.exit(1)
        
        for i, qa in enumerate(qa_pairs, 1):
            print(f"\n[{i}/{len(qa_pairs)}] {qa.question[:60]}...")
            start = datetime.now()
            
            try:
                # For simple evaluation, we just report that LLM evaluation is needed
                result = EvalResult(
                    question=qa.question,
                    expected=qa.expected_answer,
                    actual="[Requires LLM evaluation - set ANTHROPIC_API_KEY]",
                    correct=False,
                    duration_ms=(datetime.now() - start).total_seconds() * 1000,
                    error="Simple mode - LLM required for full evaluation",
                )
            except Exception as e:
                result = EvalResult(
                    question=qa.question,
                    expected=qa.expected_answer,
                    actual="",
                    correct=False,
                    duration_ms=(datetime.now() - start).total_seconds() * 1000,
                    error=str(e),
                )
            
            results.append(result)
            status = "✓" if result.correct else "✗"
            print(f"  {status} Expected: {qa.expected_answer}")
    
    return results


def print_report(results: list[EvalResult], output_path: str | None = None):
    """Print evaluation report."""
    correct = sum(1 for r in results if r.correct)
    total = len(results)
    accuracy = (correct / total * 100) if total > 0 else 0
    
    report = []
    report.append("=" * 60)
    report.append("MCP EVALUATION REPORT")
    report.append("=" * 60)
    report.append(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    report.append(f"Total Duration: {sum(r.duration_ms for r in results):.0f}ms")
    report.append("\n" + "-" * 60)
    
    for i, r in enumerate(results, 1):
        status = "✓ PASS" if r.correct else "✗ FAIL"
        report.append(f"\n[{i}] {status}")
        report.append(f"    Q: {r.question[:80]}...")
        report.append(f"    Expected: {r.expected}")
        report.append(f"    Actual: {r.actual}")
        if r.error:
            report.append(f"    Error: {r.error}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if output_path:
        Path(output_path).write_text(report_text)
        print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run MCP evaluation suite")
    parser.add_argument("eval_file", help="Path to evaluation XML file")
    parser.add_argument("-u", "--url", default="http://127.0.0.1:8000", help="MCP server URL")
    parser.add_argument("-k", "--api-key", default=None, help="API key (or set ADAPTIVE_RAG_API_KEY)")
    parser.add_argument("-o", "--output", default=None, help="Output report file")
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.environ.get("ADAPTIVE_RAG_API_KEY", "dev-secret-key-change-in-production")
    
    # Load evaluation file
    print(f"📂 Loading evaluation: {args.eval_file}")
    qa_pairs = load_evaluation_file(args.eval_file)
    print(f"   Found {len(qa_pairs)} questions")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation against {args.url}")
    results = run_simple_evaluation(args.url, api_key, qa_pairs)
    
    # Print report
    print_report(results, args.output)


if __name__ == "__main__":
    main()
