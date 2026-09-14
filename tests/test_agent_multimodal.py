"""
Tests for Phase 3 Multimodal Corrective Multi-Agent Layer (LangGraph).
Tests ContextSufficiencyEvaluator, multimodal chunk grading, inline citations,
and the LangGraph state machine execution.
"""

import pytest
from src.ingest.base import Modality, IngestedChunk
from src.retrieval.models import RetrievedChunk
from src.crag.grader import CRAGRelevanceGrader, GradingDecision
from src.crag.evaluator import ContextSufficiencyEvaluator, SufficiencyDecision
from src.generator.generator import AnswerGenerator
from src.agent.graph import run_crag_pipeline
from src.storage.vector_store import QdrantStore
from src.storage.graph_store import MultimodalGraphStore
from src.retrieval.vector_retriever import EmbeddingModelCache
from src.config import update_runtime_config


@pytest.fixture
def multimodal_chunks():
    c_pdf = RetrievedChunk(
        chunk_id="chunk-pdf-1",
        modality=Modality.PDF,
        text="Acme Corp reported third quarter revenue of $142 million, representing an 18% year-over-year increase.",
        score=0.85,
        metadata={"source_file": "q3_earnings_report.pdf", "page_number": 1},
        entities=["Acme Corp", "Q3 Revenue"]
    )
    c_img = RetrievedChunk(
        chunk_id="chunk-img-1",
        modality=Modality.IMAGE,
        text="Revenue Breakdown Chart showing Cloud at 45% and Enterprise AI at 35% of total Q3 revenue.",
        score=0.78,
        metadata={"source_file": "revenue_breakdown_chart.png"},
        entities=["Acme Corp", "Cloud", "Enterprise AI"]
    )
    c_audio = RetrievedChunk(
        chunk_id="chunk-aud-1",
        modality=Modality.AUDIO,
        text="Dr. Elena Rostova announced the official launch of Project Titan during the executive briefing.",
        score=0.82,
        metadata={"source_file": "executive_briefing.wav", "start_time": 0.0, "end_time": 15.0},
        entities=["Dr. Elena Rostova", "Project Titan"]
    )
    c_distractor = RetrievedChunk(
        chunk_id="chunk-dist-1",
        modality=Modality.TEXT,
        text="Baking powder reacts with acidic ingredients in dough to release carbon dioxide gas bubbles.",
        score=0.10,
        metadata={"source_file": "baking_guide.txt"},
        entities=["Baking powder"]
    )
    return [c_pdf, c_img, c_audio], [c_distractor]


def test_chunk_citation_formatting(multimodal_chunks):
    relevant, _ = multimodal_chunks
    c_pdf, c_img, c_audio = relevant

    assert c_pdf.format_citation() == "[q3_earnings_report.pdf, p. 1]"
    assert c_img.format_citation() == "[revenue_breakdown_chart.png, Image]"
    assert c_audio.format_citation() == "[executive_briefing.wav, 00:00 - 00:15]"


def test_context_sufficiency_evaluator(multimodal_chunks):
    relevant, distractors = multimodal_chunks
    evaluator = ContextSufficiencyEvaluator(sufficiency_threshold=0.60)
    query = "What was Acme Corp Q3 revenue?"

    # Test 1: All relevant chunks -> >= 60% -> SUFFICIENT
    res_sufficient = evaluator.evaluate(query, relevant)
    assert res_sufficient.is_sufficient is True
    assert res_sufficient.decision == SufficiencyDecision.SUFFICIENT
    assert res_sufficient.recommendation == "proceed"

    # Test 2: 1 relevant chunk + 3 distractors -> 25% < 60% -> INSUFFICIENT_AMBIGUOUS
    mixed = [relevant[0], distractors[0], distractors[0], distractors[0]]
    res_ambiguous = evaluator.evaluate(query, mixed)
    assert res_ambiguous.is_sufficient is False
    assert res_ambiguous.decision == SufficiencyDecision.INSUFFICIENT_AMBIGUOUS
    assert res_ambiguous.recommendation == "rewrite"

    # Test 3: Only distractors -> 0 relevant -> INSUFFICIENT_EMPTY
    res_empty = evaluator.evaluate(query, distractors)
    assert res_empty.is_sufficient is False
    assert res_empty.decision == SufficiencyDecision.INSUFFICIENT_EMPTY
    assert res_empty.recommendation == "fallback"


def test_multimodal_chunk_grader(multimodal_chunks):
    relevant, distractors = multimodal_chunks
    grader = CRAGRelevanceGrader(threshold_high=0.55, threshold_low=0.25)
    query = "What is the Q3 revenue of Acme Corp?"

    result = grader.grade(query, relevant)
    assert result.decision in [GradingDecision.CORRECT, GradingDecision.AMBIGUOUS]
    assert len(result.passage_scores) == 3
    assert result.relevant_count >= 1

    # Check that score objects preserve modality and title
    score0 = result.passage_scores[0]
    assert score0.modality in ["pdf", "image", "audio", "text"]
    assert score0.title != ""


def test_multimodal_generator_citations(multimodal_chunks):
    relevant, _ = multimodal_chunks
    generator = AnswerGenerator()
    query = "What was the revenue and who launched Project Titan?"

    gen_result = generator.generate(query, relevant)
    assert gen_result["answer"] != ""
    assert len(gen_result["citations"]) > 0

    # Ensure citations are formatted properly
    citations_str = " ".join(gen_result["citations"])
    assert "q3_earnings_report.pdf" in citations_str or "executive_briefing.wav" in citations_str


def test_multimodal_agent_graph_execution(multimodal_chunks):
    relevant, _ = multimodal_chunks
    # Build in-memory QdrantStore and MultimodalGraphStore
    vector_store = QdrantStore(use_memory=True, vector_dim=384)
    graph_store = MultimodalGraphStore()
    G = graph_store.get_graph()

    # Add entities and chunks to stores
    model = EmbeddingModelCache.get_model("all-MiniLM-L6-v2")
    texts = [c.text for c in relevant]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    ingested_chunks = []
    for c, emb in zip(relevant, embeddings):
        ic = IngestedChunk(
            chunk_id=c.chunk_id,
            modality=c.modality,
            text=c.text,
            metadata=c.metadata,
            entities=c.entities,
            embedding=emb
        )
        ingested_chunks.append(ic)
        for ent in c.entities:
            G.add_node(ent, modalities=[c.modality.value], source_chunk_ids=[c.chunk_id], mention_count=1)

    vector_store.add_chunks(ingested_chunks, embeddings)

    # Run CRAG pipeline with multimodal stores
    update_runtime_config({"sufficiency_threshold": 0.50, "max_retries": 1})
    state = run_crag_pipeline(
        query="What was Acme Corp revenue and who is leading Project Titan?",
        vector_store=vector_store,
        graph_store=graph_store,
        strategy="ppr"
    )

    assert state.get("final_answer") != ""
    trace = state.get("execution_trace", [])
    step_names = [s["step_name"] for s in trace]

    assert "Router" in step_names
    assert "Retriever" in step_names
    assert "Grader" in step_names
    assert "Evaluator" in step_names
    assert "Generator" in step_names
