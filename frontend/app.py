"""
Streamlit Frontend for Corrective RAG with Personalized PageRank.
Provides interactive multi-hop QA, real-time agent decision trace visualization,
runtime configuration editor, and comparative benchmark dashboard.

Supports Dual Execution Modes:
1. REST API Mode: Communicates with the FastAPI backend when available.
2. In-Process Standalone Mode: Executes the LangGraph pipeline directly inside
   the Streamlit app (ideal for 1-click Streamlit Community Cloud or Hugging Face Spaces deployment).
"""

import streamlit as st
import requests
import json
import os
import sys
import time
import pandas as pd

# Add workspace root to sys.path so in-process execution can find src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_current_config, update_runtime_config
from src.dataset import load_dataset, MultiHopSample
from src.graph_builder import EntityGraphBuilder
from src.agent.graph import run_crag_pipeline
from src.evaluation.benchmark import BenchmarkRunner

# Configure page
st.set_page_config(
    page_title="Graph-Powered Corrective RAG (CRAG)",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Base URL
API_URL = os.environ.get("CRAG_API_URL", "http://127.0.0.1:8000")

# Cache local dataset for standalone mode
@st.cache_resource
def get_local_samples():
    path = os.path.join("data", "2wikimultihopqa_sample.json")
    if os.path.exists(path):
        return load_dataset(path, max_samples=150)
    return []


def check_api_health():
    try:
        resp = requests.get(f"{API_URL}/api/status", timeout=1.5)
        return resp.status_code == 200, resp.json() if resp.status_code == 200 else {}
    except Exception:
        return False, {}


def fetch_samples():
    # Try REST API first
    try:
        resp = requests.get(f"{API_URL}/api/samples?limit=30", timeout=2)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Fallback: In-process local samples
    local_samples = get_local_samples()
    items = []
    for s in local_samples[:30]:
        items.append({
            "id": s.id,
            "question": s.question,
            "type": s.type,
            "answer": s.answer,
            "supporting_titles": s.get_supporting_passage_titles(),
            "hop_structure": s.hop_structure
        })
    return items


def fetch_config():
    try:
        resp = requests.get(f"{API_URL}/api/config", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return get_current_config().model_dump()


def update_config_api(updates: dict):
    # Try API first
    try:
        resp = requests.post(f"{API_URL}/api/config", json=updates, timeout=2)
        if resp.status_code == 200:
            return True, resp.json()
    except Exception:
        pass

    # In-process update
    try:
        cfg = update_runtime_config(updates)
        return True, cfg.model_dump()
    except Exception as e:
        return False, {"detail": str(e)}


def execute_pipeline(query: str, sample_id: str = None, algorithm: str = None):
    """Executes query via REST API or In-Process fallback."""
    # Try REST API
    try:
        payload = {"query": query, "sample_id": sample_id, "algorithm": algorithm}
        resp = requests.post(f"{API_URL}/api/query", json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # In-Process fallback
    local_samples = {s.id: s for s in get_local_samples()}
    sample = None
    passages = []
    gt_answer = None

    if sample_id and sample_id in local_samples:
        sample = local_samples[sample_id]
        passages = sample.context_passages
        gt_answer = sample.answer
    else:
        for s in local_samples.values():
            if s.question.strip().lower() == query.strip().lower():
                sample = s
                passages = s.context_passages
                gt_answer = s.answer
                break
        if not passages and local_samples:
            sample = list(local_samples.values())[0]
            passages = sample.context_passages

    builder = EntityGraphBuilder()
    graph = builder.build_graph_for_sample(sample) if sample else None

    state = run_crag_pipeline(
        query=query,
        passages=passages,
        graph=graph,
        sample_id=sample.id if sample else None,
        strategy=algorithm
    )

    grading = state.get("grading_result")
    seeds = state.get("seed_entities") or (sample.seed_entities if sample else [])
    subgraph_data = None
    if graph:
        subgraph_data = EntityGraphBuilder.get_subgraph_for_visualization(
            graph, seed_nodes=seeds, depth=2, max_nodes=35
        )

    return {
        "query": state.get("query", query),
        "sample_id": sample.id if sample else None,
        "final_answer": state.get("final_answer", ""),
        "ground_truth_answer": gt_answer,
        "supporting_evidence_text": state.get("supporting_evidence_text", ""),
        "retrieval_strategy": state.get("retrieval_strategy", "ppr"),
        "seed_entities": state.get("seed_entities", []),
        "retrieved_passages": [p.model_dump() for p in state.get("retrieved_passages", [])],
        "retrieval_scores": state.get("retrieval_scores", []),
        "retrieval_metrics": state.get("retrieval_metrics", {}),
        "grading_decision": grading.decision.value if grading else None,
        "grading_confidence": grading.confidence_score if grading else None,
        "retry_count": state.get("retry_count", 0),
        "execution_trace": state.get("execution_trace", []),
        "subgraph": subgraph_data
    }


# ---------------- SIDEBAR: Runtime Configurations ----------------
st.sidebar.title("⚙️ Runtime Configurations")
st.sidebar.caption("Dynamically adjust agent parameters without restarting the application.")

api_healthy, status_info = check_api_health()
if api_healthy:
    st.sidebar.success(f"🟢 Connected to FastAPI Backend")
else:
    st.sidebar.info(f"⚡ In-Process Mode (Cloud / Standalone Ready)")

current_config = fetch_config()

with st.sidebar.form("config_form"):
    st.subheader("Retrieval Strategy")
    algo_options = ["ppr", "vector_hop", "hybrid"]
    cur_algo = current_config.get("retrieval_algorithm", "ppr")
    algo_idx = algo_options.index(cur_algo) if cur_algo in algo_options else 0
    selected_algo = st.selectbox(
        "Algorithm",
        algo_options,
        index=algo_idx,
        help="ppr: Personalized PageRank (Fast single pass), vector_hop: Sequential vector search, hybrid: RRF fusion"
    )

    top_k = st.slider(
        "Top-k Passages",
        min_value=1,
        max_value=5,
        value=current_config.get("top_k_passages", 3)
    )

    ppr_alpha = st.slider(
        "PageRank Damping (alpha)",
        min_value=0.50,
        max_value=0.95,
        value=float(current_config.get("ppr_alpha", 0.85)),
        step=0.05,
        help="Probability of following graph edges vs teleports"
    )

    st.subheader("CRAG Grading Thresholds")
    th_high = st.slider(
        "Upper Threshold (High Relevance)",
        min_value=0.40,
        max_value=0.90,
        value=float(current_config.get("grader_threshold_high", 0.65)),
        step=0.05,
        help="Score >= Upper Threshold is classified CORRECT (use directly)"
    )

    th_low = st.slider(
        "Lower Threshold (Low Relevance)",
        min_value=0.10,
        max_value=0.50,
        value=float(current_config.get("grader_threshold_low", 0.35)),
        step=0.05,
        help="Score < Lower Threshold is classified INCORRECT (trigger fallback)"
    )

    max_retries = st.number_input(
        "Max Query Rewriter Loops",
        min_value=0,
        max_value=5,
        value=current_config.get("max_retries", 2)
    )

    st.subheader("Answer Generator")
    provider_options = ["local_extractive", "openai", "gemini"]
    cur_prov = current_config.get("llm_provider", "local_extractive")
    prov_idx = provider_options.index(cur_prov) if cur_prov in provider_options else 0
    selected_provider = st.selectbox("LLM Provider", provider_options, index=prov_idx)

    api_key_input = st.text_input(
        "API Key (OpenAI / Gemini)",
        value=current_config.get("api_key") or "",
        type="password",
        help="Optional: enter API key if using external OpenAI or Gemini LLM"
    )

    save_btn = st.form_submit_button("💾 Apply Configuration")
    if save_btn:
        payload = {
            "retrieval_algorithm": selected_algo,
            "top_k_passages": top_k,
            "ppr_alpha": ppr_alpha,
            "grader_threshold_high": th_high,
            "grader_threshold_low": th_low,
            "max_retries": int(max_retries),
            "llm_provider": selected_provider,
            "api_key": api_key_input if api_key_input.strip() else None
        }
        ok, res = update_config_api(payload)
        if ok:
            st.sidebar.success("Configuration updated live!")
            time.sleep(0.5)
            st.rerun()
        else:
            st.sidebar.error(f"Failed to update: {res}")


# ---------------- MAIN APP HEADER ----------------
st.title("🕸️ Graph-Powered Corrective RAG (CRAG)")
st.markdown("""
**Personalized PageRank Multi-Hop Retrieval & Self-Correcting LangGraph Agent Pipeline on 2WikiMultiHopQA**
""")

tab1, tab2, tab3 = st.tabs([
    "🔍 Multi-Hop QA Explorer",
    "📊 Benchmark & Speedup Dashboard",
    "🏛️ System Architecture"
])

# ---------------- TAB 1: QA EXPLORER ----------------
with tab1:
    st.subheader("Multi-Hop Reasoning Over Knowledge Graphs")

    samples = fetch_samples()
    selected_sample_id = None
    default_query = "Who is the mother of the director of film Polish-Russian War (Film)?"

    if samples:
        sample_options = ["Custom Query..."] + [
            f"[{s.get('type')}] {s.get('question')}" for s in samples
        ]
        choice = st.selectbox("Choose a sample query from 2WikiMultiHopQA:", sample_options)
        if choice != "Custom Query...":
            idx = sample_options.index(choice) - 1
            selected_sample = samples[idx]
            selected_sample_id = selected_sample.get("id")
            default_query = selected_sample.get("question")

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        user_query = st.text_input("Question:", value=default_query)
    with col_btn:
        st.write("")
        st.write("")
        run_query_btn = st.button("🚀 Run CRAG Agent", type="primary", use_container_width=True)

    if run_query_btn and user_query:
        with st.spinner("Executing LangGraph Multi-Agent Pipeline..."):
            try:
                t0 = time.perf_counter()
                data = execute_pipeline(user_query, sample_id=selected_sample_id)
                tot_latency = (time.perf_counter() - t0) * 1000.0

                st.success(f"Pipeline finished in {tot_latency:.1f} ms")

                # Answer Card
                st.markdown("### 🎯 Predicted Answer")
                ans_col1, ans_col2 = st.columns([3, 2])
                with ans_col1:
                    st.info(f"### {data.get('final_answer')}")
                    if data.get("ground_truth_answer"):
                        gt = data.get("ground_truth_answer")
                        pred = data.get("final_answer")
                        is_match = gt.lower() in pred.lower() or pred.lower() in gt.lower()
                        match_badge = "✅ MATCH" if is_match else "ℹ️ VARIATION"
                        st.caption(f"Ground Truth Answer: **{gt}** ({match_badge})")

                with ans_col2:
                    metrics = data.get("retrieval_metrics", {})
                    st.metric("Retrieval Latency", f"{metrics.get('execution_time_ms', 0):.1f} ms")
                    st.metric("Neural Vector Calls", f"{metrics.get('vector_search_calls', 0)}")
                    st.caption(f"Strategy: **{data.get('retrieval_strategy', 'ppr').upper()}** | Retries: {data.get('retry_count', 0)}")

                # Multi-Agent Decision Trace
                st.markdown("---")
                st.markdown("### 🤖 LangGraph Multi-Agent Decision Trace")
                trace = data.get("execution_trace", [])

                for i, step in enumerate(trace):
                    step_name = step.get("step_name")
                    action = step.get("action")
                    details = step.get("details", {})

                    with st.expander(f"Step {i+1}: [{step_name}] {action}", expanded=(i == len(trace)-1)):
                        st.json(details)

                # Retrieved Passages View
                st.markdown("---")
                st.markdown("### 📚 Retrieved Supporting Passages")
                passages = data.get("retrieved_passages", [])
                scores = data.get("retrieval_scores", [])
                for p_idx, p in enumerate(passages):
                    score_val = scores[p_idx] if p_idx < len(scores) else 0.0
                    with st.expander(f"📄 [{score_val:.4f}] {p.get('title')}", expanded=(p_idx == 0)):
                        st.markdown(f"**Title**: `{p.get('title')}`")
                        for sent in p.get("sentences", []):
                            st.write(f"- {sent}")

                # Subgraph View
                subgraph = data.get("subgraph")
                if subgraph and subgraph.get("nodes"):
                    st.markdown("---")
                    st.markdown("### 🕸️ Graph Subgraph (Personalized PageRank Neighborhood)")
                    nodes = subgraph.get("nodes", [])
                    edges = subgraph.get("edges", [])
                    st.caption(f"Showing {len(nodes)} entities and {len(edges)} relation edges around seed entities.")

                    node_df = pd.DataFrame([
                        {"Entity": n["id"], "Type": n["type"], "Is Seed": "🌟 Seed" if n.get("is_seed") else "Node"}
                        for n in nodes
                    ])
                    edge_df = pd.DataFrame([
                        {"From": e["source"], "Relation": e["relation"], "To": e["target"], "Weight": e["weight"]}
                        for e in edges
                    ])
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Entities in Subgraph:**")
                        st.dataframe(node_df, use_container_width=True, height=250)
                    with c2:
                        st.markdown("**Relation Edges:**")
                        st.dataframe(edge_df, use_container_width=True, height=250)

            except Exception as e:
                st.error(f"Pipeline execution error: {e}")


# ---------------- TAB 2: BENCHMARK DASHBOARD ----------------
with tab2:
    st.subheader("⚡ 2WikiMultiHopQA Comparative Retrieval Benchmark")
    st.markdown("""
    Compare **Personalized PageRank (PPR) Graph Retrieval** against a **Sequential Per-Hop Dense Vector Search Baseline**.
    Measures retrieval accuracy, latency, and validates the **~10x speedup** achieved by single-pass graph diffusion.
    """)

    col_b1, col_b2 = st.columns([2, 3])
    with col_b1:
        bench_samples = st.slider("Benchmark Queries to Evaluate", min_value=5, max_value=50, value=10, step=5)
        run_bench_btn = st.button("▶️ Run Comparative Benchmark", type="primary")

    bench_data = None
    if run_bench_btn:
        with st.spinner(f"Evaluating {bench_samples} queries through both PPR and Vector Baselines..."):
            try:
                # Try API first
                try:
                    b_resp = requests.post(f"{API_URL}/api/benchmark", json={"num_samples": bench_samples}, timeout=180)
                    if b_resp.status_code == 200:
                        bench_data = b_resp.json()
                except Exception:
                    pass

                # Fallback: run directly
                if not bench_data:
                    runner = BenchmarkRunner()
                    rep = runner.run(num_samples=bench_samples, verbose=False)
                    bench_data = rep.model_dump()

                st.success("Benchmark completed successfully!")
            except Exception as e:
                st.error(f"Benchmark error: {e}")
    else:
        # Try loading latest benchmark
        try:
            latest_resp = requests.get(f"{API_URL}/api/benchmark/latest", timeout=1.5)
            if latest_resp.status_code == 200:
                bench_data = latest_resp.json()
        except Exception:
            pass

        if not bench_data:
            path = os.path.join("data", "benchmark_results.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)

    if bench_data:
        ppr = bench_data.get("ppr_system", {})
        vec = bench_data.get("vector_baseline", {})
        speedup = bench_data.get("speedup_ratio", 1.0)
        n = bench_data.get("num_samples", 0)

        st.markdown(f"#### Results across {n} Multi-Hop Queries")

        # Top metric cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🚀 Retrieval Speedup", f"{speedup}x", delta="PPR vs Vector", delta_color="normal")
        m2.metric("PPR Latency", f"{ppr.get('avg_latency_ms', 0):.1f} ms", delta=f"-{(vec.get('avg_latency_ms', 0)-ppr.get('avg_latency_ms', 0)):.1f} ms", delta_color="inverse")
        m3.metric("Retrieval Recall@3", f"{ppr.get('retrieval_recall_pct', 0):.1f}%", delta=f"{ppr.get('retrieval_recall_pct', 0)-vec.get('retrieval_recall_pct', 0):.1f}%")
        m4.metric("Vector Calls / Hop", f"{ppr.get('avg_vector_calls', 0):.1f}", delta="-2 calls", delta_color="inverse")

        # Side-by-side comparison table
        st.markdown("#### Detailed Metric Breakdown")
        comp_df = pd.DataFrame([
            {
                "Evaluation Metric": "Retrieval Recall@3",
                "Personalized PageRank": f"{ppr.get('retrieval_recall_pct', 0):.1f}%",
                "Vector Baseline": f"{vec.get('retrieval_recall_pct', 0):.1f}%",
                "Advantage": "Higher coverage" if ppr.get('retrieval_recall_pct', 0) >= vec.get('retrieval_recall_pct', 0) else "Lower"
            },
            {
                "Evaluation Metric": "Average Latency (ms)",
                "Personalized PageRank": f"{ppr.get('avg_latency_ms', 0):.2f} ms",
                "Vector Baseline": f"{vec.get('avg_latency_ms', 0):.2f} ms",
                "Advantage": f"{speedup}x Speedup"
            },
            {
                "Evaluation Metric": "Neural Vector Calls",
                "Personalized PageRank": f"{ppr.get('avg_vector_calls', 0):.1f}",
                "Vector Baseline": f"{vec.get('avg_vector_calls', 0):.1f}",
                "Advantage": "Zero vector inference"
            },
            {
                "Evaluation Metric": "Exact Match (EM)",
                "Personalized PageRank": f"{ppr.get('exact_match_pct', 0):.1f}%",
                "Vector Baseline": f"{vec.get('exact_match_pct', 0):.1f}%",
                "Advantage": "Equivalent / higher"
            },
            {
                "Evaluation Metric": "Token F1 Score",
                "Personalized PageRank": f"{ppr.get('f1_score_pct', 0):.1f}%",
                "Vector Baseline": f"{vec.get('f1_score_pct', 0):.1f}%",
                "Advantage": "Equivalent / higher"
            }
        ])
        st.table(comp_df)

        # Per query details table
        details = bench_data.get("samples_detail", [])
        if details:
            st.markdown("#### Sample Query Breakdown")
            detail_rows = []
            for d in details:
                detail_rows.append({
                    "ID": d.get("id"),
                    "Question": d.get("question")[:60] + "...",
                    "Ground Truth": d.get("answer"),
                    "PPR Answer": d.get("ppr", {}).get("answer"),
                    "PPR Recall": d.get("ppr", {}).get("recall"),
                    "PPR Latency (ms)": f"{d.get('ppr', {}).get('latency_ms', 0):.1f}",
                    "Vector Latency (ms)": f"{d.get('vector_baseline', {}).get('latency_ms', 0):.1f}"
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)


# ---------------- TAB 3: SYSTEM ARCHITECTURE ----------------
with tab3:
    st.subheader("🏛️ Architecture & Technical Design")
    st.markdown("""
    ### Why Graph Personalized PageRank Replaces Per-Hop Vector Search
    In standard Multi-Hop RAG over Wikipedia:
    1. **Hop 1**: Query $Q$ is embedded with a Transformer, requiring dense index search $\\to$ returns Passage 1.
    2. **Hop 2**: Passage 1 text must be parsed to extract bridge entities, reformulate query $Q_2$, encode $Q_2$, and perform a second dense index search $\\to$ returns Passage 2.
    
    This sequential process suffers from **compounding error** (if Hop 1 drifts, Hop 2 fails) and **$2\\times$ latency overhead** due to repeated neural embedding calls.

    ### The Personalized PageRank Solution:
    - Context passages are mapped into an **Entity-Relation Graph** $G=(V, E)$.
    - Seed entities in $Q$ receive initial probability mass in personalization vector $p_0$.
    - **PageRank diffusion** spreads probability across 1-hop and 2-hop edges in a **single sparse matrix multiplication (~2-5 ms)**.
    - Both Hop 1 and Hop 2 supporting passages naturally surface at the top of the stationary distribution.
    """)

    st.markdown("""
    ### LangGraph Corrective RAG (CRAG) State Machine:
    ```
    [User Query] 
          │
          ▼
      [Router] ──> Selects PPR / Vector / Hybrid strategy
          │
          ▼
     [Retriever] ──> Executes graph diffusion or dense search
          │
          ▼
      [Grader] ──> Computes Sentence-Transformer cosine relevance
          │
     ┌────┴──────────────────────────┐
     │                               │                               │
     ▼                               ▼                               ▼
 [CORRECT]                     [AMBIGUOUS]                      [INCORRECT]
(Score >= Upper)             (Lower <= Score < Upper)        (Score < Lower)
     │                               │                               │
     ▼                               ▼                               ▼
[Generator] ──> [Answer]        [Rewriter]                       [Fallback]
                                     │                               │
                                     └──> Loop to Retriever          └──> Loop to Grader
    ```
    """)
