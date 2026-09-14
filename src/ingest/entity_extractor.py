"""
Entity and Knowledge Triple Extractor.
Extracts named entities and (subject, relation, object) triples from text across all modalities.
Supports structured LLM output (OpenAI / Ollama) with robust spaCy offline fallback.
"""

from typing import List, Tuple, Dict, Any, Optional
import json
import re
import spacy
from pydantic import BaseModel, Field
from src.config import get_current_config


class ExtractedEntity(BaseModel):
    name: str
    type: str = "GENERAL"


class ExtractedTriple(BaseModel):
    subject: str
    relation: str
    object: str


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    triples: List[ExtractedTriple] = Field(default_factory=list)


class EntityExtractor:
    """
    Extracts entities and relation triples from multimodal chunks.
    """

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = None
        self._spacy_model_name = spacy_model

    def _ensure_nlp(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self._spacy_model_name)
            except Exception:
                try:
                    import subprocess, sys
                    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    self.nlp = spacy.load("en_core_web_sm")
                except Exception:
                    self.nlp = spacy.blank("en")

    def extract(self, text: str) -> ExtractionResult:
        """Extracts entities and triples from text."""
        config = get_current_config()

        # Mode 1: LLM structured extraction if API key is provided
        if config.llm_provider == "openai" and config.api_key:
            llm_res = self._extract_via_llm(text, config)
            if llm_res and (llm_res.entities or llm_res.triples):
                return llm_res

        # Mode 2: spaCy NLP extractor (Deterministic offline fallback)
        return self._extract_via_spacy(text)

    def _extract_via_llm(self, text: str, config) -> Optional[ExtractionResult]:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model=config.openai_model, api_key=config.api_key, temperature=0.0)

            prompt = (
                "You are an expert knowledge graph engineer.\n"
                "Extract all named entities and factual relation triples from the following text.\n"
                "Respond ONLY with a JSON object matching this schema:\n"
                "{\n"
                '  "entities": [{"name": "string", "type": "string"}],\n'
                '  "triples": [{"subject": "string", "relation": "string", "object": "string"}]\n'
                "}\n\n"
                f"Text:\n{text[:1500]}\n"
            )

            resp = llm.invoke(prompt)
            raw = resp.content.strip()
            # Clean possible markdown block
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            parsed = json.loads(raw)
            entities = [ExtractedEntity(**e) for e in parsed.get("entities", [])]
            triples = [ExtractedTriple(**t) for t in parsed.get("triples", [])]
            return ExtractionResult(entities=entities, triples=triples)
        except Exception:
            return None

    def _extract_via_spacy(self, text: str) -> ExtractionResult:
        self._ensure_nlp()
        if not self.nlp or not text.strip():
            return ExtractionResult()

        doc = self.nlp(text[:2000])
        entities: List[ExtractedEntity] = []
        seen_ents = set()

        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) > 1 and not name.isdigit() and name.lower() not in seen_ents:
                seen_ents.add(name.lower())
                entities.append(ExtractedEntity(name=name, type=ent.label_))

        # Heuristic triple extraction based on syntactic dependencies
        triples: List[ExtractedTriple] = []
        for sent in doc.sents:
            subjects = [tok for tok in sent if "subj" in tok.dep_]
            objects = [tok for tok in sent if "obj" in tok.dep_]

            for s in subjects:
                for o in objects:
                    verb = s.head
                    rel_text = verb.lemma_ if verb.pos_ == "VERB" else "relates_to"
                    s_str = s.text.strip()
                    o_str = o.text.strip()
                    if s_str and o_str and s_str.lower() != o_str.lower():
                        triples.append(ExtractedTriple(subject=s_str, relation=rel_text, object=o_str))

        return ExtractionResult(entities=entities, triples=triples[:10])
