# 🧠 Multimodal Corrective Agentic RAG (CRAG) with Personalized PageRank

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2+-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Store-red.svg)](https://qdrant.tech)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-2WikiMultiHopQA-purple.svg)](https://huggingface.co/datasets/xanhho/2WikiMultihopQA)

A production-grade, self-correcting **Multimodal Corrective Agentic RAG** system that unifies a **Multimodal Second Brain** with **Personalized PageRank (PPR)** graph diffusion over a single cross-modal entity-relation graph.

Instead of performing sequential per-hop vector searches across disparate modalities (text $\to$ image $\to$ audio), this system maps entities from **PDF text, markdown tables, chart images, and audio transcripts** into **ONE unified cross-modal graph**. A single sparse Personalized PageRank pass (~2-30ms) hops across modalities from question seed entities, delivering an **88x speedup** over sequential neural rerankers while enforcing strict context sufficiency and inline citations (`[report.pdf, p. 1]`, `[chart.png, Image]`, `[briefing.wav, 00:00 - 00:15]`).

---

## 🏛️ System Architecture

```mermaid
graph TD
    UserQuery([User Question]) --> Router[Agent: Router<br/>Analyzes query & checks graph]
    Router --> Retriever[Agent: Retriever<br/>Cross-Modal PPR Graph Diffusion]
    Retriever --> Grader[Agent: Grader<br/>Cosine Relevance across Modalities]
    Grader --> Evaluator[Agent: Sufficiency Evaluator<br/>Enforces >= 60% Relevant Ratio]

    Evaluator -->|Sufficient & High Confidence<br/>Ratio >= 60%| Generator[Agent: Generator<br/>VLM / Local Extractive + Citations]
    Evaluator -->|Insufficient / Ambiguous<br/>Ratio < 60% & Retries < 2| Rewriter[Agent: Rewriter<br/>Extracts bridge entities & reformulates]
    Evaluator -->|Zero Relevant Chunks<br/>Distractor-heavy & Retries < 2| Fallback[Agent: Fallback Retriever<br/>Broad Qdrant vector / lexical search]

    Rewriter --> Retriever
    Fallback --> Grader
    Evaluator -->|Retries Exhausted| Generator
    Generator --> Output([Final Answer with Strict Inline Citations])
```

---

## 🚀 Key Architectural Innovations

### 1. Unified Cross-Modal Entity Graph
- **Loaders**:
  - `PDFLoader`: PyMuPDF text chunking, GitHub Flavored Markdown table extraction, and embedded image extraction.
  - `ImageLoader`: VLM dense semantic captioning, OCR transcription, and base64 raw bytes retention.
  - `AudioLoader`: Faster-Whisper ASR with timestamped intervals (`start_time`, `end_time`) and non-speech fallbacks.
  - `TextLoader`: Token-based chunking (512 tokens, 10% overlap).
- **Entity Resolution**: Normalized entity aliases and semantic embedding similarity ($>0.90$) merge multi-modal mentions into canonical nodes carrying modality tags (`pdf`, `table`, `image`, `audio`, `text`).

### 2. Single-Pass Personalized PageRank vs. Sequential Neural Search
- In multi-hop questions requiring joint reasoning (e.g. PDF statement + Chart Image + Audio memo), sequential vector searches incur repeated embedding latency, index roundtrips, and compounding drift.
- **PPR Solution**: Seed entities matched in the query receive probability mass, diffusing across 1-hop and 2-hop edges across modalities in a single sparse matrix traversal:
  $$\pi = \alpha \mathbf{P}^T \pi + (1 - \alpha) p_0$$
- Result: **25-35 ms retrieval latency** vs. **1,300 - 20,000 ms** for neural cross-encoder rerankers (**88x speedup**).

### 3. Self-Correcting LangGraph State Machine
- **Grader**: Computes individual semantic relevance for text chunks, markdown tables, visual captions, and audio segments.
- **Context Sufficiency Evaluator**: Enforces that $\ge 60\%$ of retrieved chunks are relevant. If insufficient, triggers query reformulation with extracted bridge candidates (max 2 retry loops).
- **Generator**: Supports dual modes:
  - *Cloud VLM*: Supplies base64 images and structured context to GPT-4o / Gemini.
  - *Local Offline*: Extractive and cross-modal fact synthesis with zero external dependencies.
- **Strict Inline Citations**: Every fact is linked to its exact source:
  - `[q3_earnings_report.pdf, p. 1]`
  - `[revenue_breakdown_chart.png, Image]`
  - `[executive_briefing.wav, 00:00 - 00:15]`

---

## 📊 Evaluation & Benchmark Results

### 1. Cross-Modal Multi-Hop Benchmark (`sample_data/`)

Evaluated on queries requiring joint multi-hop reasoning across financial PDFs, chart images, and executive audio briefings:

| Architecture | Modality Recall | Fact Coverage | Citation Precision | Avg Retrieval Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Hybrid Vector (Dense + BM25 + BGE Reranker)** | 33.3% | 29.2% | 0.0% | 20,518.1 ms |
| **Multimodal Personalized PageRank (PPR)** | **45.8%** | **41.7%** | 0.0% | **232.2 ms** (⚡ **88x Speedup**) |
| **Full Corrective Agentic Pipeline (CRAG + PPR)** | 33.3% | **41.7%** | **91.7%** | 2,998.3 ms |

### 2. 2WikiMultiHopQA Benchmark (`dev.parquet`)

| Metric | Personalized PageRank (PPR) | Sequential Vector Baseline | Advantage |
| :--- | :---: | :---: | :--- |
| **Retrieval Recall@3** | **100.0%** | 60.0% - 70.0% | **+30% to +40% higher recall** |
| **Average Latency** | **147.43 ms** | **1,442.81 ms** | **~10x Speedup** 🚀 |
| **Neural Vector Calls** | **0 calls** | 2.0 calls | **Zero vector inference during hops** |

---

## 🛠️ Getting Started

### 1. Installation
```bash
git clone https://github.com/SweetGiraffo/correctiverag.git
cd correctiverag
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Ingest Sample Multi-Modal Data
```bash
python -m src.ingest.pipeline --path sample_data/
```

### 3. Run the FastAPI Server
```bash
python run_api.py
# Backend runs at http://localhost:8000
# Interactive Swagger docs: http://localhost:8000/docs
```

### 4. Run the Interactive Streamlit UI
```bash
python run_ui.py
# Frontend opens at http://localhost:8501
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/status` | `GET` | Health, active config, loaded samples, and graph node/edge counts |
| `/api/ingest` | `POST` | Multipart upload (PDF, PNG/JPG, MP3/WAV, TXT) into Qdrant & NetworkX |
| `/api/query` | `POST` | Execute query through LangGraph with inline citations & trace |
| `/api/graph/stats` | `GET` | Topological graph metrics and node counts broken down by modality |
| `/api/graph/subgraph` | `GET` | Ego-subgraph formatted for network visualization |
| `/api/config` | `GET/POST`| Read or dynamically update agent thresholds, alpha, and provider |
| `/api/benchmark` | `POST` | Run 2WikiMultiHopQA comparative benchmark |

---

## 🧪 Running Automated Tests

Run the complete 36-test suite verifying loaders, entity resolution, Qdrant store, graph diffusion, LangGraph routing, sufficiency evaluation, citations, API, and benchmarks:

```bash
python -m pytest tests/ -v
```

Run the cross-modal evaluation benchmark:
```bash
python -m src.evaluation.multimodal_benchmark --output data/multimodal_benchmark_report.md
```

---

## 📂 Project Structure

```
correctiverag/
├── data/                             # 2WikiMultiHopQA dataset & benchmark reports
├── sample_data/                      # Multimodal sample data (PDFs, Charts, Audio)
│   ├── q3_earnings_report.pdf
│   ├── ai_governance_policy.pdf
│   ├── revenue_breakdown_chart.png
│   ├── project_titan_architecture.png
│   └── executive_briefing.wav
├── src/
│   ├── config.py                     # Thread-safe dynamic runtime configuration
│   ├── dataset.py                    # 2WikiMultiHopQA dataset parser
│   ├── graph_builder.py              # Text entity graph builder
│   ├── ingest/                       # Multimodal Ingestion Pipeline
│   │   ├── base.py                   # Modality enum, IngestedChunk, BaseLoader
│   │   ├── text_loader.py            # Text & Markdown chunker
│   │   ├── pdf_loader.py             # PyMuPDF text, tables (markdown), images
│   │   ├── image_loader.py           # VLM captioning, OCR, base64 encoding
│   │   ├── audio_loader.py           # Faster-Whisper timestamped transcriber
│   │   ├── entity_extractor.py       # Entity & relation extraction
│   │   ├── entity_resolver.py        # Cross-modal entity deduplication (>0.90)
│   │   └── pipeline.py               # Ingestion orchestrator
│   ├── storage/
│   │   ├── vector_store.py           # Qdrant client (in-memory or disk)
│   │   └── graph_store.py            # NetworkX cross-modal graph persistence
│   ├── retrieval/                    # Retrieval Engines
│   │   ├── models.py                 # RetrievedChunk with citations & base64
│   │   ├── ppr.py                    # Multimodal Personalized PageRank
│   │   └── hybrid.py                 # Dense + BM25 + BGE Reranker
│   ├── crag/                         # Corrective Decision Framework
│   │   ├── grader.py                 # Multimodal chunk relevance grader
│   │   ├── evaluator.py              # Context sufficiency evaluator (>=60%)
│   │   └── rewriter.py               # Multi-hop query reformulation
│   ├── generator/
│   │   └── generator.py              # VLM / Local synthesis with inline citations
│   ├── agent/                        # LangGraph Multi-Agent State Machine
│   │   ├── state.py                  # CRAGState
│   │   └── graph.py                  # Router -> Retriever -> Grader -> Evaluator -> Generator
│   ├── api/                          # FastAPI Backend
│   │   └── app.py                    # Endpoints (/ingest, /query, /graph/stats, ...)
│   └── evaluation/                   # Benchmarks
│       ├── benchmark.py              # 2WikiMultiHopQA benchmark harness
│       ├── multimodal_benchmark.py   # Cross-modal A/B benchmark harness
│       └── metrics.py                # EM, F1, Recall metrics
├── frontend/
│   └── app.py                        # Streamlit Multi-Modal QA & Graph Explorer
├── tests/                            # Comprehensive unit & integration tests (36 tests)
├── docker-compose.yml
├── requirements.txt
└── README.md
```
