"""
Benchmark Runner: Evaluates PPR-based CRAG vs. Baseline Per-Hop Vector Search.
Measures retrieval recall, exact match accuracy, per-hop latency, and speedup ratio.
"""

from typing import List, Dict, Any, Optional
import sys
import time
import json
import os
import argparse
import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.dataset import load_dataset, MultiHopSample
from src.graph_builder import EntityGraphBuilder
from src.retrieval.ppr_retriever import PPRRetriever
from src.retrieval.vector_retriever import SequentialVectorRetriever
from src.generator.generator import AnswerGenerator
from src.evaluation.metrics import compute_exact_match, compute_f1, compute_retrieval_metrics
from src.config import get_current_config


class SystemBenchmarkStats(BaseModel):
    name: str
    exact_match_pct: float
    f1_score_pct: float
    retrieval_recall_pct: float
    avg_latency_ms: float
    avg_vector_calls: float


class BenchmarkReport(BaseModel):
    num_samples: int
    ppr_system: SystemBenchmarkStats
    vector_baseline: SystemBenchmarkStats
    speedup_ratio: float
    samples_detail: List[Dict[str, Any]] = []


class BenchmarkRunner:
    """
    Executes comparative benchmarks between Personalized PageRank and Sequential Vector Search.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        config = get_current_config()
        self.dataset_path = dataset_path or (
            config.sample_dataset_path
            if os.path.exists(config.sample_dataset_path)
            else config.dataset_path
        )
        self.graph_builder = EntityGraphBuilder()
        self.ppr_retriever = PPRRetriever(alpha=config.ppr_alpha)
        self.vector_retriever = SequentialVectorRetriever(model_name=config.embedding_model_name)
        self.generator = AnswerGenerator()

    def run(self, num_samples: int = 25, verbose: bool = False) -> BenchmarkReport:
        samples = load_dataset(self.dataset_path, max_samples=num_samples)
        if not samples:
            raise ValueError(f"No samples loaded from {self.dataset_path}")

        actual_n = len(samples)
        ppr_ems, ppr_f1s, ppr_recalls, ppr_latencies, ppr_vec_calls = [], [], [], [], []
        vec_ems, vec_f1s, vec_recalls, vec_latencies, vec_vec_calls = [], [], [], [], []
        details = []

        iterator = tqdm(samples, desc=f"Benchmarking {actual_n} queries") if verbose else samples

        for s in iterator:
            gt_titles = s.get_supporting_passage_titles()
            gt_answer = s.answer

            # 1. Build entity graph
            t0_graph = time.perf_counter()
            G = self.graph_builder.build_graph_for_sample(s)
            graph_time_ms = (time.perf_counter() - t0_graph) * 1000.0

            # 2. Evaluate Personalized PageRank System
            ppr_res = self.ppr_retriever.retrieve(
                query=s.question,
                graph=G,
                passages=s.context_passages,
                top_k=3
            )
            ppr_titles = [p.title for p in ppr_res.passages]
            ppr_ret_metrics = compute_retrieval_metrics(ppr_titles, gt_titles)
            ppr_gen = self.generator.generate(s.question, ppr_res.passages)
            ppr_em = compute_exact_match(ppr_gen["answer"], gt_answer)
            ppr_f1 = compute_f1(ppr_gen["answer"], gt_answer)

            # Total latency includes retrieval
            ppr_total_latency = ppr_res.execution_time_ms
            ppr_ems.append(ppr_em)
            ppr_f1s.append(ppr_f1)
            ppr_recalls.append(ppr_ret_metrics["recall"])
            ppr_latencies.append(ppr_total_latency)
            ppr_vec_calls.append(ppr_res.vector_search_calls)

            # 3. Evaluate Baseline Sequential Per-Hop Vector Search
            vec_res = self.vector_retriever.retrieve(
                query=s.question,
                passages=s.context_passages,
                top_k=3,
                hops=2
            )
            vec_titles = [p.title for p in vec_res.passages]
            vec_ret_metrics = compute_retrieval_metrics(vec_titles, gt_titles)
            vec_gen = self.generator.generate(s.question, vec_res.passages)
            vec_em = compute_exact_match(vec_gen["answer"], gt_answer)
            vec_f1 = compute_f1(vec_gen["answer"], gt_answer)

            vec_total_latency = vec_res.execution_time_ms
            vec_ems.append(vec_em)
            vec_f1s.append(vec_f1)
            vec_recalls.append(vec_ret_metrics["recall"])
            vec_latencies.append(vec_total_latency)
            vec_vec_calls.append(vec_res.vector_search_calls)

            details.append({
                "id": s.id,
                "question": s.question,
                "answer": gt_answer,
                "ppr": {
                    "answer": ppr_gen["answer"],
                    "em": ppr_em,
                    "f1": round(ppr_f1, 4),
                    "recall": ppr_ret_metrics["recall"],
                    "latency_ms": ppr_total_latency,
                    "retrieved_titles": ppr_titles
                },
                "vector_baseline": {
                    "answer": vec_gen["answer"],
                    "em": vec_em,
                    "f1": round(vec_f1, 4),
                    "recall": vec_ret_metrics["recall"],
                    "latency_ms": vec_total_latency,
                    "retrieved_titles": vec_titles
                }
            })

        avg_ppr_lat = float(np.mean(ppr_latencies)) if ppr_latencies else 1.0
        avg_vec_lat = float(np.mean(vec_latencies)) if vec_latencies else 1.0
        # Avoid division by zero
        speedup = round(avg_vec_lat / max(avg_ppr_lat, 0.01), 2)

        ppr_stats = SystemBenchmarkStats(
            name="Personalized PageRank (PPR)",
            exact_match_pct=round(float(np.mean(ppr_ems)) * 100.0, 2),
            f1_score_pct=round(float(np.mean(ppr_f1s)) * 100.0, 2),
            retrieval_recall_pct=round(float(np.mean(ppr_recalls)) * 100.0, 2),
            avg_latency_ms=round(avg_ppr_lat, 2),
            avg_vector_calls=round(float(np.mean(ppr_vec_calls)), 1)
        )

        vec_stats = SystemBenchmarkStats(
            name="Baseline Per-Hop Vector Search",
            exact_match_pct=round(float(np.mean(vec_ems)) * 100.0, 2),
            f1_score_pct=round(float(np.mean(vec_f1s)) * 100.0, 2),
            retrieval_recall_pct=round(float(np.mean(vec_recalls)) * 100.0, 2),
            avg_latency_ms=round(avg_vec_lat, 2),
            avg_vector_calls=round(float(np.mean(vec_vec_calls)), 1)
        )

        return BenchmarkReport(
            num_samples=actual_n,
            ppr_system=ppr_stats,
            vector_baseline=vec_stats,
            speedup_ratio=speedup,
            samples_detail=details
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 2WikiMultiHopQA comparative retrieval benchmark")
    parser.add_argument("--samples", type=int, default=20, help="Number of queries to benchmark")
    parser.add_argument("--output", type=str, default="data/benchmark_results.json", help="Path to save output JSON")
    args = parser.parse_args()

    print(f"Starting benchmark over {args.samples} multi-hop queries...")
    runner = BenchmarkRunner()
    report = runner.run(num_samples=args.samples, verbose=True)

    print("\n" + "=" * 65)
    print("                2WikiMultiHopQA BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Total Evaluated Queries: {report.num_samples}\n")
    print(f"{'Metric':<25} | {'Personalized PageRank':<20} | {'Vector Baseline'}")
    print("-" * 65)
    print(f"{'Retrieval Recall@3':<25} | {report.ppr_system.retrieval_recall_pct:.1f}%{'':<15} | {report.vector_baseline.retrieval_recall_pct:.1f}%")
    print(f"{'Exact Match (EM)':<25} | {report.ppr_system.exact_match_pct:.1f}%{'':<15} | {report.vector_baseline.exact_match_pct:.1f}%")
    print(f"{'F1 Score':<25} | {report.ppr_system.f1_score_pct:.1f}%{'':<15} | {report.vector_baseline.f1_score_pct:.1f}%")
    print(f"{'Average Latency':<25} | {report.ppr_system.avg_latency_ms:.2f} ms{'':<13} | {report.vector_baseline.avg_latency_ms:.2f} ms")
    print(f"{'Vector Search Calls':<25} | {report.ppr_system.avg_vector_calls:.1f}{'':<18} | {report.vector_baseline.avg_vector_calls:.1f}")
    print("=" * 65)
    print(f">> Retrieval Speedup: {report.speedup_ratio}x faster than baseline vector search!\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    print(f"Detailed benchmark report saved to: {args.output}")
