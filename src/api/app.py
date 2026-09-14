"""
FastAPI Backend Application for Corrective RAG with Personalized PageRank.
Exposes endpoints for multimodal querying, ingestion, graph topology stats,
dynamic runtime configuration, sample exploration, and benchmarking.
"""

from typing import List, Dict, Any, Optional
import os
import shutil
import tempfile
import json
from fastapi import FastAPI, HTTPException, Body, UploadFile, File, Form
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
from src.retrieval.models import RetrievedChunk
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.ingest.pipeline import IngestionPipeline


app = FastAPI(
    title="Multimodal Corrective RAG with Personalized PageRank API",
    description="Multi-Agent CRAG Pipeline over Cross-Modal Entity Graph (Text, PDF Tables, Images, Audio) and 2WikiMultiHopQA",
    version="2.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stores and pipeline singletons
_dataset_cache: Dict[str, MultiHopSample] = {}
_graph_builder = EntityGraphBuilder()
_vector_store: Optional[QdrantStore] = None
_graph_store: Optional[MultimodalGraphStore] = None
_ingestion_pipeline: Optional[IngestionPipeline] = None


def get_stores() -> tuple[QdrantStore, MultimodalGraphStore, IngestionPipeline]:
    global _vector_store, _graph_store, _ingestion_pipeline
    if _graph_store is None:
        _graph_store = MultimodalGraphStore()
    if _vector_store is None:
        _vector_store = QdrantStore(vector_dim=384)
    if _ingestion_pipeline is None:
        _ingestion_pipeline = IngestionPipeline(vector_store=_vector_store, graph_store=_graph_store)
    return _vector_store, _graph_store, _ingestion_pipeline


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
    query: str = Field(..., description="Multi-hop or multimodal question text")
    sample_id: Optional[str] = Field(None, description="Optional ID of existing 2WikiMultiHopQA sample")
    algorithm: Optional[str] = Field(None, description="Override retrieval algorithm ('ppr', 'vector_hop', 'hybrid')")
    custom_passages: Optional[List[Passage]] = Field(None, description="Optional custom candidate passages")
    use_multimodal: bool = Field(True, description="Whether to prioritize the unified multimodal graph and vector store")


class QueryResponse(BaseModel):
    query: str
    sample_id: Optional[str]
    final_answer: str
    ground_truth_answer: Optional[str] = None
    supporting_evidence_text: str
    retrieval_strategy: str
    seed_entities: List[str]
    retrieved_passages: List[Passage]
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    retrieval_scores: List[float]
    retrieval_metrics: Dict[str, Any]
    grading_decision: Optional[str] = None
    grading_confidence: Optional[float] = None
    sufficiency_decision: Optional[str] = None
    relevant_ratio: Optional[float] = None
    retry_count: int
    execution_trace: List[Dict[str, Any]]
    subgraph: Optional[Dict[str, Any]] = None


class BenchmarkRequest(BaseModel):
    num_samples: int = Field(default=10, ge=1, le=100, description="Number of queries to benchmark")


# API Endpoints
@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Returns system status, active configuration, dataset, and graph availability."""
    config = get_current_config()
    samples = get_cached_samples()
    _, g_store, _ = get_stores()
    dev_exists = os.path.exists(config.dataset_path)
    sample_exists = os.path.exists(config.sample_dataset_path)
    g_stats = g_store.get_stats()

    return {
        "status": "online",
        "loaded_samples_count": len(samples),
        "dev_dataset_present": dev_exists,
        "sample_dataset_present": sample_exists,
        "multimodal_graph_nodes": g_stats["total_nodes"],
        "multimodal_graph_edges": g_stats["total_edges"],
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


@app.post("/api/ingest")
async def ingest_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Uploads and ingests a document (PDF, PNG/JPG, MP3/WAV, TXT/MD) into Qdrant & NetworkX."""
    v_store, g_store, pipeline = get_stores()
    suffix = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        chunks = pipeline.ingest_file(tmp_path)
        # Update metadata filename
        for c in chunks:
            c.metadata["source_file"] = file.filename
        stats = g_store.get_stats()
        return {
            "status": "success",
            "file_name": file.filename,
            "chunks_count": len(chunks),
            "entities_extracted": sum(len(c.entities) for c in chunks),
            "modalities": list(dict.fromkeys(c.modality.value for c in chunks)),
            "graph_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/api/graph/stats")
def get_graph_stats() -> Dict[str, Any]:
    """Returns topological statistics across modalities for the unified entity graph."""
    _, g_store, _ = get_stores()
    return g_store.get_stats()


@app.get("/api/graph/subgraph")
def get_graph_subgraph(seeds: Optional[str] = None, max_nodes: int = 40) -> Dict[str, Any]:
    """Extracts an ego-subgraph formatted for PyVis / network visualization."""
    _, g_store, _ = get_stores()
    seed_list = [s.strip() for s in seeds.split(",")] if seeds else []
    return g_store.get_subgraph(seeds=seed_list, max_nodes=max_nodes)


@app.post("/api/query", response_model=QueryResponse)
def execute_query(req: QueryRequest) -> QueryResponse:
    """Executes a query through the LangGraph Corrective RAG pipeline."""
    samples = get_cached_samples()
    v_store, g_store, _ = get_stores()
    graph_is_populated = len(g_store.get_graph()) > 0

    # Decision: Run Multimodal Pipeline or Dataset Mode
    run_multimodal = req.use_multimodal and graph_is_populated and not req.sample_id and not req.custom_passages

    if run_multimodal:
        state = run_crag_pipeline(
            query=req.query,
            vector_store=v_store,
            graph_store=g_store,
            strategy=req.algorithm
        )
        grading = state.get("grading_result")
        eval_res = state.get("evaluation_result")
        seeds = state.get("seed_entities", [])
        subgraph_data = g_store.get_subgraph(seeds=seeds, max_nodes=35)

        return QueryResponse(
            query=state.get("query", req.query),
            sample_id=None,
            final_answer=state.get("final_answer", ""),
            ground_truth_answer=None,
            supporting_evidence_text=state.get("supporting_evidence_text", ""),
            retrieval_strategy=state.get("retrieval_strategy", "ppr"),
            seed_entities=seeds,
            retrieved_passages=state.get("retrieved_passages", []),
            retrieved_chunks=state.get("retrieved_chunks", []),
            citations=state.get("citations", []),
            retrieval_scores=state.get("retrieval_scores", []),
            retrieval_metrics=state.get("retrieval_metrics", {}),
            grading_decision=grading.decision.value if grading else None,
            grading_confidence=grading.confidence_score if grading else None,
            sufficiency_decision=eval_res.decision.value if eval_res else None,
            relevant_ratio=eval_res.relevant_ratio if eval_res else None,
            retry_count=state.get("retry_count", 0),
            execution_trace=state.get("execution_trace", []),
            subgraph=subgraph_data
        )

    # Legacy / Dataset Mode
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

        # Fallback to first sample
        if not passages and samples:
            sample = list(samples.values())[0]
            passages = sample.context_passages

    if not passages:
        raise HTTPException(status_code=400, detail="No passages or multimodal content available for retrieval.")

    state = run_crag_pipeline(
        query=req.query,
        passages=passages,
        graph=graph,
        sample_id=sample.id if sample else None,
        strategy=req.algorithm
    )

    grading = state.get("grading_result")
    eval_res = state.get("evaluation_result")

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
        retrieved_chunks=state.get("retrieved_chunks", []),
        citations=state.get("citations", []),
        retrieval_scores=state.get("retrieval_scores", []),
        retrieval_metrics=state.get("retrieval_metrics", {}),
        grading_decision=grading.decision.value if grading else None,
        grading_confidence=grading.confidence_score if grading else None,
        sufficiency_decision=eval_res.decision.value if eval_res else None,
        relevant_ratio=eval_res.relevant_ratio if eval_res else None,
        retry_count=state.get("retry_count", 0),
        execution_trace=state.get("execution_trace", []),
        subgraph=subgraph_data
    )


@app.post("/api/benchmark", response_model=BenchmarkReport)
def run_benchmark_endpoint(req: BenchmarkRequest) -> BenchmarkReport:
    """Runs a comparative benchmark over the dataset and returns the report."""
    runner = BenchmarkRunner()
    report = runner.run(num_samples=req.num_samples, verbose=False)
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
