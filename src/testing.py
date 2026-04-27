"""
Phase 7: Testing & Evaluation Suite
Comprehensive test suite for the RAG pipeline.
"""

import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# =============================================================================
# TEST QUERIES
# =============================================================================

TEST_CASES = [
    # Test Suite 1: Retrieval Accuracy
    {"id": "T-001", "query": "What is an agent?", "expected": "definition"},
    {"id": "T-002", "query": "Explain Q-Learning", "expected": "explanation"},
    {"id": "T-003", "query": "What is xyz123notfound?", "expected": "refusal"},
    {"id": "T-004", "query": "What is quantum physics?", "expected": "refusal"},
    {"id": "T-005", "query": "Types of agents", "expected": "list"},
    
    # Test Suite 2: Topic Coverage
    {"id": "T-006", "query": "Markov Decision Process", "expected": "mdp"},
    {"id": "T-007", "query": "Reinforcement Learning process", "expected": "rl"},
    {"id": "T-008", "query": "Sensor and Actuator", "expected": "sensor"},
    {"id": "T-009", "query": "Scaling planning", "expected": "planning"},
    {"id": "T-010", "query": "Rewards in RL", "expected": "rewards"},
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_retrieval(vectorstore, test_cases: List[Dict], k: int = 3) -> Dict:
    """
    Evaluate retrieval quality for test queries.
    
    Returns:
        Report with retrieval metrics
    """
    results = []
    total_time = 0
    
    for case in test_cases:
        start = time.time()
        docs = vectorstore.similarity_search(case["query"], k=k)
        elapsed = time.time() - start
        total_time += elapsed
        
        has_results = len(docs) > 0
        has_metadata = all(
            d.metadata.get("source_file") and d.metadata.get("page_number")
            for d in docs
        ) if docs else False
        
        results.append({
            "id": case["id"],
            "query": case["query"],
            "results_found": len(docs),
            "has_metadata": has_metadata,
            "response_time": round(elapsed, 3),
            "sources": [f"{d.metadata.get('source_file', 'N/A')} p.{d.metadata.get('page_number', 'N/A')}" 
                       for d in docs[:2]],
        })
    
    avg_time = total_time / len(test_cases) if test_cases else 0
    
    report = {
        "total_tests": len(test_cases),
        "avg_response_time": round(avg_time, 3),
        "all_have_results": all(r["results_found"] > 0 for r in results),
        "all_have_metadata": all(r["has_metadata"] for r in results),
        "individual_results": results,
    }
    
    report["passed"] = report["all_have_results"] and report["all_have_metadata"]
    
    return report


def evaluate_rag_chain(chain, test_cases: List[Dict]) -> Dict:
    """
    Evaluate the full RAG chain (retrieval + generation).
    
    Returns:
        Report with RAG metrics
    """
    from src.llm_chain import post_process_answer
    
    results = []
    total_time = 0
    
    for case in test_cases:
        start = time.time()
        answer = chain(case["query"])
        elapsed = time.time() - start
        total_time += elapsed
        
        processed = post_process_answer(answer)
        
        results.append({
            "id": case["id"],
            "query": case["query"],
            "answer": processed["answer"][:200],
            "has_citation": len(processed["sources"]) > 0,
            "confidence": processed["confidence"],
            "response_time": round(elapsed, 3),
        })
    
    avg_time = total_time / len(test_cases) if test_cases else 0
    citation_rate = sum(r["has_citation"] for r in results) / len(results)
    
    report = {
        "total_tests": len(test_cases),
        "avg_response_time": round(avg_time, 3),
        "citation_rate": round(citation_rate, 2),
        "all_responded": all(len(r["answer"]) > 10 for r in results),
        "individual_results": results,
    }
    
    report["passed"] = report["citation_rate"] >= 0.8 and report["avg_response_time"] < 5.0
    
    return report


def run_full_evaluation(vectorstore, chain) -> Dict:
    """
    Run complete evaluation suite.
    
    Returns:
        Combined evaluation report
    """
    logger.info("Running full evaluation suite...")
    
    # Phase 1: Retrieval evaluation
    retrieval_report = evaluate_retrieval(vectorstore, TEST_CASES)
    
    # Phase 2: Full RAG evaluation
    rag_report = evaluate_rag_chain(chain, TEST_CASES)
    
    combined = {
        "retrieval": retrieval_report,
        "rag": rag_report,
        "overall_passed": retrieval_report["passed"] and rag_report["passed"],
    }
    
    return combined


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def print_report(report: Dict):
    """Pretty print evaluation report."""
    print("=" * 60)
    print("📊 EVALUATION REPORT")
    print("=" * 60)
    
    # Retrieval
    r = report.get("retrieval", {})
    print(f"\n🔍 RETRIEVAL EVALUATION")
    print(f"   Total tests: {r.get('total_tests', 0)}")
    print(f"   Avg response time: {r.get('avg_response_time', 0):.3f}s")
    print(f"   All have results: {r.get('all_have_results', False)}")
    print(f"   All have metadata: {r.get('all_have_metadata', False)}")
    
    # RAG
    rag = report.get("rag", {})
    print(f"\n🤖 RAG EVALUATION")
    print(f"   Total tests: {rag.get('total_tests', 0)}")
    print(f"   Avg response time: {rag.get('avg_response_time', 0):.3f}s")
    print(f"   Citation rate: {rag.get('citation_rate', 0):.0%}")
    print(f"   All responded: {rag.get('all_responded', False)}")
    
    # Overall
    print(f"\n{'✅ OVERALL: PASSED' if report.get('overall_passed') else '❌ OVERALL: FAILED'}")
    print("=" * 60)
