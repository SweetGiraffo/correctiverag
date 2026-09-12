"""
Unit tests for 2WikiMultiHopQA dataset loading and parsing.
"""

import os
import pytest
from src.dataset import load_dataset, MultiHopSample, Passage, EvidenceTriple


def test_load_sample_dataset():
    sample_path = os.path.join("data", "2wikimultihopqa_sample.json")
    assert os.path.exists(sample_path), "Sample dataset json file should exist"

    samples = load_dataset(sample_path, max_samples=10)
    assert len(samples) == 10
    for s in samples:
        assert isinstance(s, MultiHopSample)
        assert s.id != ""
        assert s.question != ""
        assert s.answer != ""
        assert len(s.context_passages) > 0
        assert len(s.supporting_facts) > 0
        assert len(s.evidences) > 0
        assert len(s.seed_entities) > 0
        assert len(s.hop_structure) > 0


def test_passage_and_evidence_structure():
    sample_path = os.path.join("data", "2wikimultihopqa_sample.json")
    samples = load_dataset(sample_path, max_samples=5)
    sample = samples[0]

    # Test supporting passage titles
    sup_titles = sample.get_supporting_passage_titles()
    assert len(sup_titles) > 0
    all_titles = [p.title for p in sample.context_passages]
    for t in sup_titles:
        assert t in all_titles

    # Test supporting text consolidation
    sup_text = sample.get_supporting_text()
    assert len(sup_text) > 0
    assert any(t in sup_text for t in sup_titles)
