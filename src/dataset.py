"""
Dataset loader and parser for 2WikiMultiHopQA.
Parses multi-hop questions, context passages, supporting facts, and evidence hop chains.
"""

from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import os
import pandas as pd


class Passage(BaseModel):
    """Represents a candidate context passage/document."""
    title: str
    sentences: List[str]
    text: str

    @classmethod
    def from_raw(cls, raw: Any) -> "Passage":
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            title = str(raw[0])
            sents = [str(s) for s in raw[1]] if isinstance(raw[1], (list, tuple)) else [str(raw[1])]
            text = " ".join(sents)
            return cls(title=title, sentences=sents, text=text)
        elif isinstance(raw, dict):
            title = raw.get("title", "")
            sents = raw.get("sentences", [])
            text = raw.get("text", " ".join(sents))
            return cls(title=title, sentences=sents, text=text)
        else:
            return cls(title="Unknown", sentences=[str(raw)], text=str(raw))


class EvidenceTriple(BaseModel):
    """An annotated entity-relation-entity triple representing a reasoning step."""
    subject: str
    relation: str
    object: str

    def to_string(self) -> str:
        return f"({self.subject}) -[{self.relation}]-> ({self.object})"


class MultiHopSample(BaseModel):
    """Complete representation of a 2WikiMultiHopQA query."""
    id: str
    question: str
    type: str
    answer: str
    supporting_facts: List[Tuple[str, int]] = Field(default_factory=list)
    evidences: List[EvidenceTriple] = Field(default_factory=list)
    context_passages: List[Passage] = Field(default_factory=list)
    hop_structure: List[str] = Field(default_factory=list)
    seed_entities: List[str] = Field(default_factory=list)
    bridge_entities: List[str] = Field(default_factory=list)

    def get_supporting_passage_titles(self) -> List[str]:
        """Returns distinct titles of passages that contain supporting facts."""
        return list(dict.fromkeys(title for title, _ in self.supporting_facts))

    def get_supporting_text(self) -> str:
        """Constructs a consolidated string of the exact supporting sentences."""
        passages_by_title = {p.title: p for p in self.context_passages}
        supporting_lines = []
        for title, sent_idx in self.supporting_facts:
            passage = passages_by_title.get(title)
            if passage and 0 <= sent_idx < len(passage.sentences):
                supporting_lines.append(f"[{title}] {passage.sentences[sent_idx]}")
            else:
                supporting_lines.append(f"[{title}] (Sentence {sent_idx})")
        return "\n".join(supporting_lines)


def parse_raw_field(field_val: Any) -> Any:
    """Helper to parse JSON strings or return python objects as-is."""
    if isinstance(field_val, str):
        try:
            return json.loads(field_val)
        except Exception:
            return field_val
    return field_val


def parse_multihop_sample(row: Dict[str, Any]) -> MultiHopSample:
    """Converts a raw dictionary/dataframe row into a typed MultiHopSample."""
    sample_id = str(row.get("_id", row.get("id", "")))
    question = str(row.get("question", "")).strip()
    q_type = str(row.get("type", "unknown"))
    answer = str(row.get("answer", "")).strip()

    # Parse supporting facts
    raw_sp = parse_raw_field(row.get("supporting_facts", []))
    supporting_facts: List[Tuple[str, int]] = []
    if isinstance(raw_sp, (list, tuple)):
        for item in raw_sp:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    supporting_facts.append((str(item[0]), int(item[1])))
                except (ValueError, TypeError):
                    continue

    # Parse evidence triples
    raw_ev = parse_raw_field(row.get("evidences", []))
    evidences: List[EvidenceTriple] = []
    if isinstance(raw_ev, (list, tuple)):
        for item in raw_ev:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                evidences.append(
                    EvidenceTriple(
                        subject=str(item[0]).strip(),
                        relation=str(item[1]).strip(),
                        object=str(item[2]).strip()
                    )
                )

    # Parse context passages
    raw_ctx = parse_raw_field(row.get("context", []))
    context_passages: List[Passage] = []
    if isinstance(raw_ctx, (list, tuple)):
        for p in raw_ctx:
            context_passages.append(Passage.from_raw(p))

    # Compute hop structure, seed entities, bridge entities
    hop_structure: List[str] = []
    seed_entities: List[str] = []
    bridge_entities: List[str] = []

    if evidences:
        for idx, ev in enumerate(evidences):
            hop_structure.append(f"Hop {idx + 1}: {ev.to_string()}")

        subjects = [ev.subject for ev in evidences]
        objects = [ev.object for ev in evidences]

        # Seed entity is typically the starting subject in hop 1
        if subjects:
            seed_entities.append(subjects[0])

        # In multi-hop, an entity that appears as an object in hop 1 and subject in hop 2 is a bridge
        for obj in objects:
            if obj in subjects[1:]:
                bridge_entities.append(obj)
    else:
        # Fallback: extract title of first supporting fact as seed
        if supporting_facts:
            seed_entities.append(supporting_facts[0][0])

    return MultiHopSample(
        id=sample_id,
        question=question,
        type=q_type,
        answer=answer,
        supporting_facts=supporting_facts,
        evidences=evidences,
        context_passages=context_passages,
        hop_structure=hop_structure,
        seed_entities=list(dict.fromkeys(seed_entities)),
        bridge_entities=list(dict.fromkeys(bridge_entities))
    )


def load_dataset(filepath: str, max_samples: Optional[int] = None) -> List[MultiHopSample]:
    """
    Loads samples from Parquet or JSON file.
    If the file is not found, raises FileNotFoundError.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    samples: List[MultiHopSample] = []
    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
        if max_samples:
            df = df.iloc[:max_samples]
        for _, row in df.iterrows():
            samples.append(parse_multihop_sample(row.to_dict()))
    elif filepath.endswith(".json") or filepath.endswith(".jsonl"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                if max_samples:
                    data = data[:max_samples]
                for item in data:
                    samples.append(parse_multihop_sample(item))
            elif isinstance(data, dict):
                samples.append(parse_multihop_sample(data))

    return samples
