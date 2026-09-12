"""
Answer Generator supporting dual modes:
1. Fast local extractive / semantic reader (offline, deterministic, zero-cost).
2. LLM generation (OpenAI / Gemini) when API keys are configured.
"""

from typing import List, Dict, Any, Optional
import re
import spacy
from src.dataset import Passage
from src.config import RuntimeConfig, get_current_config


class AnswerGenerator:
    """
    Synthesizes answers from retrieved context.
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

    def generate(
        self,
        query: str,
        retrieved_passages: List[Passage],
        config: Optional[RuntimeConfig] = None
    ) -> Dict[str, str]:
        """
        Generates answer and returns dictionary:
        {"answer": str, "supporting_text": str, "mode": str}
        """
        active_config = config or get_current_config()

        if not retrieved_passages:
            return {
                "answer": "Unable to determine answer; no relevant passages found.",
                "supporting_text": "",
                "mode": "fallback"
            }

        # Format context string
        context_blocks = []
        for p in retrieved_passages:
            sents = p.sentences[:4] if p.sentences else [p.text[:300]]
            context_blocks.append(f"Passage: {p.title}\n" + " ".join(sents))
        full_context = "\n\n".join(context_blocks)

        # Mode 1: If external LLM is configured with an active key
        if active_config.llm_provider in ["openai", "gemini"] and active_config.api_key:
            llm_answer = self._generate_with_llm(query, full_context, active_config)
            if llm_answer:
                return {
                    "answer": llm_answer,
                    "supporting_text": full_context,
                    "mode": f"llm_{active_config.llm_provider}"
                }

        # Mode 2: Local extractive / rule-based reader
        extractive_answer = self._extract_answer_local(query, retrieved_passages)
        return {
            "answer": extractive_answer,
            "supporting_text": full_context,
            "mode": "local_extractive"
        }

    def _generate_with_llm(self, query: str, context: str, config: RuntimeConfig) -> Optional[str]:
        try:
            prompt = (
                "You are an accurate multi-hop question answering system.\n"
                "Answer the following question based ONLY on the provided context passages.\n"
                "Be direct, concise, and provide the exact answer entity or statement.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )

            if config.llm_provider == "openai":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=config.openai_model,
                    api_key=config.api_key,
                    temperature=0.0
                )
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

    def _extract_answer_local(self, query: str, passages: List[Passage]) -> str:
        """
        Extractive entity reader using spaCy NER and target relation cues.
        """
        self._ensure_nlp()
        q_lower = query.lower()

        # Identify target entity type requested
        target_label = None
        if re.search(r"\b(who|mother|father|director|actor|writer|composer|singer|founder|performer|spouse|husband|wife|daughter|son)\b", q_lower):
            target_label = "PERSON"
        elif re.search(r"\b(where|country|city|state|birthplace|headquarters|location|born in)\b", q_lower):
            target_label = "GPE"
        elif re.search(r"\b(when|year|date|born|died|release|released|founded)\b", q_lower):
            target_label = "DATE"

        # Check for comparison question: "Which [film/book/etc.], X or Y, was [attribute]?"
        comp_or_match = re.search(r"\b(?:which|what)\b.+?,?\s+(.+?)\s+or\s+(.+?)\??$", query, re.IGNORECASE)
        if comp_or_match:
            opt1 = comp_or_match.group(1).strip().rstrip(",?")
            opt2 = comp_or_match.group(2).strip().rstrip("?")
            # Score which option matches positive attribute in text
            t1_count = sum(p.text.count(opt1) for p in passages)
            t2_count = sum(p.text.count(opt2) for p in passages)
            if t1_count > t2_count:
                return opt1
            elif t2_count > t1_count:
                return opt2

        # Extract entities from the top 2 passages (especially the 2nd hop passage)
        candidate_entities = []
        for p in passages[:3]:
            # Focus on sentences that contain target keywords or relation words
            for sent in p.sentences:
                doc = self.nlp(sent)
                for ent in doc.ents:
                    ent_text = ent.text.strip()
                    # Disqualify entities that are already mentioned in the question text
                    if ent_text.lower() in q_lower:
                        continue
                    if len(ent_text) <= 1 or ent_text.isdigit():
                        continue

                    score = 1.0
                    if target_label and ent.label_ == target_label:
                        score += 3.0
                    # Bonus if the entity is not the passage title itself (likely the answer inside the passage)
                    if ent_text != p.title:
                        score += 1.5

                    candidate_entities.append((ent_text, score))

        if candidate_entities:
            # Sort by score descending
            candidate_entities.sort(key=lambda x: x[1], reverse=True)
            return candidate_entities[0][0]

        # Fallback: return title of top retrieved passage that wasn't in the question
        for p in passages:
            if p.title.lower() not in q_lower:
                return p.title

        return passages[0].title if passages else "No answer found."
