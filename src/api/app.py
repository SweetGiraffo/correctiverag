"""
FastAPI Backend Application for Corrective RAG with Personalized PageRank.
Exposes endpoints for querying, pipeline status, dynamic runtime configuration,
samples exploration, and benchmarking.
"""

from typing import List, Dict, Any, Optional
import os
import json
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    RuntimeConfig,
    get_current_config,
    update_runtime_config
)
from src.dataset import load_dataset, MultiHopSample, Passage
from src.graph_builder import EntityGraphBuilder
from src.agent.graph import run_crag_pipeline
from src.evaluation.benchmark import BenchmarkRunner, BenchmarkReport


app = FastAPI(
    title="Corrective RAG with Personalized PageRank API",
    description="Multi-Agent CRAG Pipeline over 2WikiMultiHopQA with Graph-Based Retrieval",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory dataset cache for fast request routing
_dataset_cache: Dict[str, MultiHopSample] = {}
_graph_builder = EntityGraphBuilder()


def get_cached_samples() -> Dict[str, MultiHopSample]:
    global _dataset_cache
    if not _dataset_cache:
        config = get_current_config()
        path = config.sample_dataset_path if os.path.exists(config.sample_dataset_path) else config.dataset_path
        if os.path.exists(path):
            samples = load_dataset(path, max_samples=150)
            _dataset_cache = {s.id: s for s in samples}
    return _dataset_cache


# Request / Response Schemas
class QueryRequest(BaseModel):
    query: str = Field(..., description="Multi-hop question text")
    sample_id: Optional[str] = Field(None, description="Optional ID of existing 2WikiMultiHopQA sample")
    algorithm: Optional[str] = Field(None, description="Override retrieval algorithm ('ppr', 'vector_hop', 'hybrid')")
    custom_passages: Optional[List[Passage]] = Field(None, description="Optional custom candidate passages")


class QueryResponse(BaseModel):
    query: str
    sample_id: Optional[str]
    final_answer: str
    ground_truth_answer: Optional[str]
    supporting_evidence_text: str
    retrieval_strategy: str
    seed_entities: List[str]
    retrieved_passages: List[Passage]
    retrieval_scores: List[float]
    retrieval_metrics: Dict[str, Any]
    grading_decision: Optional[str]
    grading_confidence: Optional[float]
    retry_count: int
    execution_trace: List[Dict[str, Any]]
    subgraph: Optional[Dict[str, Any]] = None


class BenchmarkRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=100, description="Number of queries to benchmark")


# API Endpoints
@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns system status, active configuration, and dataset availability."""
    config = get_current_config()
    samples = get_cached_samples()
    dev_exists = os.path.exists(config.dataset_path)
    sample_exists = os.path.exists(config.sample_dataset_path)

    return {
        "status": "online",
        "loaded_samples_count": len(samples),
        "dev_dataset_present": dev_exists,
        "sample_dataset_present": sample_exists,
        "config": config.model_dump()
    }


@app.get("/api/config", response_model=RuntimeConfig)
def get_config() -> RuntimeConfig:
    """Returns the current runtime configuration."""
    return get_current_config()


@app.post("/api/config", response_model=RuntimeConfig)
def update_config(updates: Dict[str, Any] = Body(...)) -> RuntimeConfig:
    """Updates runtime configuration settings on the fly without restarting."""
    try:
        new_config = update_runtime_config(updates)
        return new_config
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/samples")
def get_samples(limit: int = 50) -> List[Dict[str, Any]]:
    """Lists preloaded 2WikiMultiHopQA sample queries for one-click testing."""
    samples = get_cached_samples()
    items = []
    for s in list(samples.values())[:limit]:
        items.append({
            "id": s.id,
            "question": s.question,
            "type": s.type,
            "answer": s.answer,
            "supporting_titles": s.get_supporting_passage_titles(),
            "hop_structure": s.hop_structure
        })
    return items


@app.post("/api/query", response_model=QueryResponse)
def execute_query(req: QueryRequest) -> QueryResponse:
    """Executes a query through the LangGraph Corrective RAG pipeline."""
    samples = get_cached_samples()
    sample: Optional[MultiHopSample] = None
    passages: List[Passage] = []
    graph = None
    gt_answer = None

    if req.sample_id and req.sample_id in samples:
        sample = samples[req.sample_id]
        passages = sample.context_passages
        gt_answer = sample.answer
        graph = _graph_builder.build_graph_for_sample(sample)
    elif req.custom_passages:
        passages = req.custom_passages
    else:
        # Check if query matches a known sample question
        for s in samples.values():
            if s.question.strip().lower() == req.query.strip().lower():
                sample = s
                passages = s.context_passages
                gt_answer = s.answer
                graph = _graph_builder.build_graph_for_sample(s)
                break

        # If still no passages, fallback to first sample's passages
        if not passages and samples:
            sample = list(samples.values())[0]
            passages = sample.context_passages

    if not passages:
        raise HTTPException(status_code=400, detail="No passages available for retrieval.")

    # Execute LangGraph Pipeline
    state = run_crag_pipeline(
        query=req.query,
        passages=passages,
        graph=graph,
        sample_id=sample.id if sample else None,
        strategy=req.algorithm
    )

    grading = state.get("grading_result")

    # Subgraph for visualization
    subgraph_data = None
    active_graph = state.get("graph") or graph
    if active_graph and len(active_graph) > 0:
        seeds = state.get("seed_entities") or (sample.seed_entities if sample else [])
        subgraph_data = EntityGraphBuilder.get_subgraph_for_visualization(
            active_graph,
            seed_nodes=seeds,
            depth=2,
            max_nodes=35
        )

    return QueryResponse(
        query=state.get("query", req.query),
        sample_id=sample.id if sample else None,
        final_answer=state.get("final_answer", ""),
        ground_truth_answer=gt_answer,
        supporting_evidence_text=state.get("supporting_evidence_text", ""),
        retrieval_strategy=state.get("retrieval_strategy", "ppr"),
        seed_entities=state.get("seed_entities", []),
        retrieved_passages=state.get("retrieved_passages", []),
        retrieval_scores=state.get("retrieval_scores", []),
        retrieval_metrics=state.get("retrieval_metrics", {}),
        grading_decision=grading.decision.value if grading else None,
        grading_confidence=grading.confidence_score if grading else None,
        retry_count=state.get("retry_count", 0),
        execution_trace=state.get("execution_trace", []),
        subgraph=subgraph_data
    )


@app.post("/api/benchmark", response_model=BenchmarkReport)
def run_benchmark_endpoint(req: BenchmarkRequest) -> BenchmarkReport:
    """Runs a comparative benchmark over the dataset and returns the report."""
    runner = BenchmarkRunner()
    report = runner.run(num_samples=req.num_samples, verbose=False)
    # Save results
    os.makedirs("data", exist_ok=True)
    with open("data/benchmark_results.json", "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return report


@app.get("/api/benchmark/latest")
def get_latest_benchmark() -> Dict[str, Any]:
    """Retrieves the latest saved benchmark report."""
    path = "data/benchmark_results.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="No previous benchmark run found.")
