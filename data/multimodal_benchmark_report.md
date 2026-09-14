# 📊 Cross-Modal Multi-Hop RAG Benchmark Report

**Evaluated on 4 cross-modal multi-hop test queries requiring joint reasoning across PDF, Chart Image, and Audio.**

## 🏆 Comparative Performance Summary

| Architecture | Modality Recall | Fact Coverage | Citation Precision | Avg Latency |
|---|---|---|---|---|
| **Hybrid Vector Search (Dense + BM25)** | 33.3% | 29.2% | 0.0% | 20518.1 ms |
| **Multimodal Personalized PageRank (PPR)** | 45.8% | 41.7% | 0.0% | 232.2 ms |
| **Full Corrective Agentic Pipeline (CRAG + PPR)** | 33.3% | 41.7% | 91.7% | 2998.3 ms |

## 🔍 Query-by-Query Analysis

### Query 1: What was Acme Corp's Q3 revenue and what share was from Cloud?
*Multi-hop reasoning linking financial PDF statement with revenue breakdown chart image.*

- **Target Modalities**: pdf, image
- **Target Facts**: 142, 45%

| Metric | Hybrid Vector | PPR Graph | Full CRAG Agent |
|---|---|---|---|
| Modalities Retrieved | pdf, text | pdf, text | pdf |
| Facts Captured | 0/2 | 0/2 | 0/2 |
| Citations Present | 0 | 0 | 2 |
| Latency | 78141.0 ms | 832.1 ms | 1543.0 ms |

> **Synthesized Agent Answer**: ACME CORP Q3 2026 FINANCIAL REPORT Executive Summary: Acme Corp reported record quarterly performance in Q3 2026, driven primarily by strong adoption of enterprise AI solutions developed by the QuantumAI Division. Quarterly Division Breakdown Table: Division | Q3 Revenue | YoY Growth | Operating Margin ----------------------------------------------------------- QuantumAI | $450M | +42% | 38% CloudScale | $320M | +18% | 24% Hardware | $180M | -5% | 12% Total company revenue reached $950M, surpassing analyst estimates by $65M. [q3_earnings_report.pdf] ACME CORP - AI GOVERNANCE AND SAFETY STANDARDS (2026) Scope and Governance Framework: This policy governs all autonomous algorithms deployed across Acme Corp and its subsidiaries. The AI Safety Council is chaired jointly by Dr. Elena Rostova and the Chief Risk Officer. Audit Requirements: All models under Project Titan must undergo adversarial stress testing before deployment on CloudScale data centers located in North America and Europe. [ai_governance_policy.pdf]

### Query 2: Who leads Project Titan and what was the quarterly revenue growth rate?
*Multi-hop bridging spoken executive audio memo with earnings report PDF.*

- **Target Modalities**: audio, pdf
- **Target Facts**: Elena Rostova, 18%

| Metric | Hybrid Vector | PPR Graph | Full CRAG Agent |
|---|---|---|---|
| Modalities Retrieved | image, pdf, text | pdf, text | pdf |
| Facts Captured | 1/2 | 2/2 | 2/2 |
| Citations Present | 0 | 0 | 2 |
| Latency | 1311.7 ms | 29.3 ms | 4371.4 ms |

> **Synthesized Agent Answer**: ACME CORP Q3 2026 FINANCIAL REPORT Executive Summary: Acme Corp reported record quarterly performance in Q3 2026, driven primarily by strong adoption of enterprise AI solutions developed by the QuantumAI Division. Quarterly Division Breakdown Table: Division | Q3 Revenue | YoY Growth | Operating Margin ----------------------------------------------------------- QuantumAI | $450M | +42% | 38% CloudScale | $320M | +18% | 24% Hardware | $180M | -5% | 12% Total company revenue reached $950M, surpassing analyst estimates by $65M. [q3_earnings_report.pdf] ACME CORP - AI GOVERNANCE AND SAFETY STANDARDS (2026) Scope and Governance Framework: This policy governs all autonomous algorithms deployed across Acme Corp and its subsidiaries. The AI Safety Council is chaired jointly by Dr. Elena Rostova and the Chief Risk Officer. Audit Requirements: All models under Project Titan must undergo adversarial stress testing before deployment on CloudScale data centers located in North America and Europe. [ai_governance_policy.pdf]

### Query 3: What are the revenue shares of Cloud and Enterprise AI, and who is leading the AI division?
*Multi-hop reasoning bridging chart image and audio announcement.*

- **Target Modalities**: image, audio
- **Target Facts**: 45%, 35%, Elena Rostova

| Metric | Hybrid Vector | PPR Graph | Full CRAG Agent |
|---|---|---|---|
| Modalities Retrieved | pdf, text | pdf, image | pdf |
| Facts Captured | 1/3 | 1/3 | 1/3 |
| Citations Present | 0 | 0 | 2 |
| Latency | 1281.7 ms | 35.3 ms | 4693.6 ms |

> **Synthesized Agent Answer**: ACME CORP Q3 2026 FINANCIAL REPORT Executive Summary: Acme Corp reported record quarterly performance in Q3 2026, driven primarily by strong adoption of enterprise AI solutions developed by the QuantumAI Division. Quarterly Division Breakdown Table: Division | Q3 Revenue | YoY Growth | Operating Margin ----------------------------------------------------------- QuantumAI | $450M | +42% | 38% CloudScale | $320M | +18% | 24% Hardware | $180M | -5% | 12% Total company revenue reached $950M, surpassing analyst estimates by $65M. [q3_earnings_report.pdf] ACME CORP - AI GOVERNANCE AND SAFETY STANDARDS (2026) Scope and Governance Framework: This policy governs all autonomous algorithms deployed across Acme Corp and its subsidiaries. The AI Safety Council is chaired jointly by Dr. Elena Rostova and the Chief Risk Officer. Audit Requirements: All models under Project Titan must undergo adversarial stress testing before deployment on CloudScale data centers located in North America and Europe. [ai_governance_policy.pdf]

### Query 4: Synthesize Acme Corp's Q3 revenue, top driver shares, and the leadership update for Project Titan.
*3-way cross-modal synthesis across PDF document, visual chart image, and audio briefing.*

- **Target Modalities**: pdf, image, audio
- **Target Facts**: 142, 45%, Titan

| Metric | Hybrid Vector | PPR Graph | Full CRAG Agent |
|---|---|---|---|
| Modalities Retrieved | pdf, text | pdf, text | pdf |
| Facts Captured | 1/3 | 1/3 | 1/3 |
| Citations Present | 0 | 0 | 2 |
| Latency | 1337.8 ms | 32.2 ms | 1385.0 ms |

> **Synthesized Agent Answer**: STRATEGIC INITIATIVES & LEADERSHIP Key Milestone: Dr. Elena Rostova was officially appointed Chief Technology Officer of the QuantumAI Division. Under her leadership, the division completed beta testing of Project Titan, a next-generation neural reasoning engine integrated into Acme Corp's CloudScale infrastructure. Capital Allocation: Acme Corp allocated $120M in research funding specifically to Project Titan for expansion into APAC. [q3_earnings_report.pdf] ACME CORP - AI GOVERNANCE AND SAFETY STANDARDS (2026) Scope and Governance Framework: This policy governs all autonomous algorithms deployed across Acme Corp and its subsidiaries. The AI Safety Council is chaired jointly by Dr. Elena Rostova and the Chief Risk Officer. Audit Requirements: All models under Project Titan must undergo adversarial stress testing before deployment on CloudScale data centers located in North America and Europe. [ai_governance_policy.pdf]
