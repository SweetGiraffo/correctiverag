"""
Answer Generator supporting dual modes:
1. Fast local extractive / cross-modal reader (offline, deterministic, zero-cost) with inline citations.
2. VLM / LLM generation (OpenAI GPT-4o / Gemini) with base64 image delivery and inline citations.
"""

from typing import List, Dict, Any, Optional, Union
import re
import spacy
from src.dataset import Passage
from src.config import RuntimeConfig, get_current_config
from src.retrieval.models import RetrievedChunk
from src.ingest.base import Modality


class AnswerGenerator:
    """
    Synthesizes multimodal answers with inline citations from retrieved context.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self._spacy_model_name = spacy_model

    def _ensure_nlp(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self._spacy_model_name, disable=["parser"])
            except Exception:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")

    def _format_chunk_context(self, item: Any) -> tuple[str, str, Optional[str]]:
        """
        Formats context string, citation, and optional base64 image for an item.
        Returns: (context_str, citation_str, image_base64)
        """
        if isinstance(item, RetrievedChunk):
            citation = item.format_citation()
            source = item.metadata.get("source_file", "Document")
            modality_str = item.modality.value.upper()
            context = f"[{modality_str} from {source} | Citation: {citation}]\n{item.text}"
            img_b64 = item.image_base64 if item.is_visual() else None
            return context, citation, img_b64
        elif isinstance(item, Passage):
            citation = f"[{item.title}]"
            sents = item.sentences[:4] if item.sentences else [item.text[:300]]
            context = f"[Passage: {item.title} | Citation: {citation}]\n" + " ".join(sents)
            return context, citation, None
        else:
            title = getattr(item, "title", str(item))
            citation = f"[{title}]"
            text = getattr(item, "text", str(item))
            context = f"[Context: {title} | Citation: {citation}]\n{text}"
            return context, citation, None

    def generate(
        self,
        query: str,
        retrieved_passages: List[Any],
        config: Optional[RuntimeConfig] = None
    ) -> Dict[str, Any]:
        """
        Generates answer and returns dictionary:
        {
            "answer": str,
            "supporting_text": str,
            "citations": List[str],
            "mode": str
        }
        """
        active_config = config or get_current_config()

        if not retrieved_passages:
            return {
                "answer": "Unable to determine answer; no relevant passages or multimodal evidence found.",
                "supporting_text": "",
                "citations": [],
                "mode": "fallback"
            }

        context_blocks = []
        citations = []
        images_base64 = []

        for item in retrieved_passages:
            block, cit, img_b64 = self._format_chunk_context(item)
            context_blocks.append(block)
            citations.append(cit)
            if img_b64:
                images_base64.append(img_b64)

        full_context = "\n\n".join(context_blocks)
        unique_citations = list(dict.fromkeys(citations))

        # Mode 1: External LLM/VLM if configured
        if active_config.llm_provider in ["openai", "gemini"] and active_config.api_key:
            llm_answer = self._generate_with_vlm(
                query=query,
                context=full_context,
                images_base64=images_base64,
                config=active_config
            )
            if llm_answer:
                return {
                    "answer": llm_answer,
                    "supporting_text": full_context,
                    "citations": unique_citations,
                    "mode": f"vlm_{active_config.llm_provider}"
                }

        # Mode 2: Local extractive / cross-modal reader
        extractive_answer = self._extract_answer_local(query, retrieved_passages)
        return {
            "answer": extractive_answer,
            "supporting_text": full_context,
            "citations": unique_citations,
            "mode": "local_extractive"
        }

    def _generate_with_vlm(
        self,
        query: str,
        context: str,
        images_base64: List[str],
        config: RuntimeConfig
    ) -> Optional[str]:
        """
        Calls external VLM (OpenAI GPT-4o or Gemini) with text context and optional images.
        """
        system_instructions = (
            "You are an accurate multimodal question answering system.\n"
            "Answer the question based strictly on the provided context passages, tables, and images.\n"
            "Crucial Rule: You MUST provide inline citations in square brackets for every fact or claim "
            "(e.g., '[q3_earnings_report.pdf, p. 1]', '[revenue_breakdown_chart.png, Image]', "
            "'[executive_briefing.wav, 00:00 - 00:05]').\n"
            "Be direct, concise, and synthesize facts across modalities where needed."
        )
        prompt = (
            f"{system_instructions}\n\n"
            f"Context Evidence:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer with citations:"
        )

        try:
            if config.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                from langchain_core.messages import HumanMessage

                llm = ChatOpenAI(
                    model=config.openai_model,
                    api_key=config.api_key,
                    temperature=0.0
                )

                if images_base64:
                    message_content = [{"type": "text", "text": prompt}]
                    for b64 in images_base64[:3]:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        })
                    response = llm.invoke([HumanMessage(content=message_content)])
                else:
                    response = llm.invoke(prompt)

                return response.content.strip()

            elif config.llm_provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=config.gemini_model,
                    google_api_key=config.api_key,
                    temperature=0.0
                )
                response = llm.invoke(prompt)
                return response.content.strip()

        except Exception:
            return None
        return None

    def _extract_answer_local(self, query: str, passages: List[Any]) -> str:
        """
        Extractive entity & cross-modal reader using spaCy NER, keywords, and citations.
        """
        self._ensure_nlp()
        q_lower = query.lower()

        # Check if items are multimodal RetrievedChunks
        is_multimodal = any(isinstance(p, RetrievedChunk) for p in passages)

        # Check for comparison question: "Which [film/book/etc.], X or Y, was [attribute]?"
        comp_or_match = re.search(r"\b(?:which|what)\b.+?,?\s+(.+?)\s+or\s+(.+?)\??$", query, re.IGNORECASE)
        if comp_or_match:
            opt1 = comp_or_match.group(1).strip().rstrip(",?")
            opt2 = comp_or_match.group(2).strip().rstrip("?")
            t1_count = sum(getattr(p, "text", "").count(opt1) for p in passages)
            t2_count = sum(getattr(p, "text", "").count(opt2) for p in passages)
            if t1_count > t2_count:
                return opt1
            elif t2_count > t1_count:
                return opt2

        # Multimodal handling: Extract facts and pair with citations
        if is_multimodal:
            return self._extract_multimodal_synthesis(query, passages)

        # Standard 2Wiki entity reader
        target_label = None
        if re.search(r"\b(who|mother|father|director|actor|writer|composer|singer|founder|performer|spouse|husband|wife|daughter|son)\b", q_lower):
            target_label = "PERSON"
        elif re.search(r"\b(where|country|city|state|birthplace|headquarters|location|born in)\b", q_lower):
            target_label = "GPE"
        elif re.search(r"\b(when|year|date|born|died|release|released|founded)\b", q_lower):
            target_label = "DATE"

        candidate_entities = []
        for p in passages[:3]:
            sentences = p.sentences if hasattr(p, "sentences") and p.sentences else [getattr(p, "text", "")]
            title = getattr(p, "title", "")

            for sent in sentences:
                doc = self.nlp(sent)
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    if ent_text.lower() in q_lower:
                        continue
                    if len(ent_text) <= 1 or ent_text.isdigit():
                        continue

                    score = 1.0
                    if target_label and ent.label_ == target_label:
                        score += 3.0
                    if ent_text != title:
                        score += 1.5

                    candidate_entities.append((ent_text, score))

        if candidate_entities:
            candidate_entities.sort(key=lambda x: x[1], reverse=True)
            return candidate_entities[0][0]

        # Fallback: title of top retrieved passage
        for p in passages:
            t = getattr(p, "title", "")
            if t and t.lower() not in q_lower:
                return t

        return getattr(passages[0], "title", "No answer found.") if passages else "No answer found."

    def _extract_multimodal_synthesis(self, query: str, chunks: List[Any]) -> str:
        """
        Synthesizes answers across multimodal chunks with explicit inline citations.
        """
        q_words = set(re.findall(r"\w+", query.lower())) - {"what", "who", "when", "where", "which", "how", "the", "and", "is", "was", "are", "were"}
        extracted_facts = []

        for c in chunks:
            if not isinstance(c, RetrievedChunk):
                continue
            citation = c.format_citation()
            text = c.text

            # Look for lines or sentences that overlap with question keywords
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for line in lines:
                line_words = set(re.findall(r"\w+", line.lower()))
                overlap = len(q_words & line_words)
                if overlap > 0:
                    # Clean markdown formatting
                    clean_line = line.lstrip("-*#| ")
                    if len(clean_line) > 15:
                        extracted_facts.append((clean_line, citation, overlap))

        if extracted_facts:
            # Sort by keyword overlap descending
            extracted_facts.sort(key=lambda x: x[2], reverse=True)
            # Take top distinct facts
            selected = []
            seen_cits = set()
            for fact, cit, _ in extracted_facts:
                if cit not in seen_cits and len(selected) < 3:
                    selected.append(f"{fact} {cit}")
                    seen_cits.add(cit)

            if selected:
                return " ".join(selected)

        # Fallback: summarize top chunk with citation
        top_c = chunks[0]
        cit = top_c.format_citation() if hasattr(top_c, "format_citation") else f"[{getattr(top_c, 'title', 'Document')}]"
        first_line = top_c.text.split("\n")[0][:200]
        return f"{first_line} {cit}"
