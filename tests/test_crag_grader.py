"""
Unit tests for CRAG Relevance Grader and Query Rewriter.
"""

import pytest
from src.dataset import Passage
from src.crag.grader import CRAGRelevanceGrader, GradingDecision
from src.crag.rewriter import CRAGQueryRewriter


@pytest.fixture
def sample_passages():
    p1 = Passage(
        title="Polish-Russian War (film)",
        sentences=[
            "Polish-Russian War is a 2009 Polish drama film directed by Xawery Żuławski.",
            "It is based on the novel Snow White and Russian Red by Dorota Masłowska."
        ],
        text="Polish-Russian War is a 2009 Polish drama film directed by Xawery Żuławski. It is based on the novel Snow White and Russian Red by Dorota Masłowska."
    )
    p2 = Passage(
        title="Xawery Żuławski",
        sentences=[
            "Xawery Żuławski is a Polish film director born in Warsaw in 1971.",
            "He is the son of actress Małgorzata Braunek and director Andrzej Żuławski."
        ],
        text="Xawery Żuławski is a Polish film director born in Warsaw in 1971. He is the son of actress Małgorzata Braunek and director Andrzej Żuławski."
    )
    p_distractor = Passage(
        title="Baking Powder",
        sentences=[
            "Baking powder is a dry chemical leavening agent used for increasing the volume of baked goods."
        ],
        text="Baking powder is a dry chemical leavening agent used for increasing the volume of baked goods."
    )
    return [p1, p2], [p_distractor]


def test_crag_grader_correct(sample_passages):
    relevant_passages, _ = sample_passages
    query = "Who is the mother of the director of film Polish-Russian War?"

    grader = CRAGRelevanceGrader(threshold_high=0.55, threshold_low=0.25)
    result = grader.grade(query, relevant_passages)

    assert result.decision in [GradingDecision.CORRECT, GradingDecision.AMBIGUOUS]
    assert result.confidence_score > 0.35
    assert len(result.passage_scores) == 2
    assert result.passage_scores[0].title in ["Polish-Russian War (film)", "Xawery Żuławski"]


def test_crag_grader_incorrect(sample_passages):
    _, distractors = sample_passages
    query = "Who is the mother of the director of film Polish-Russian War?"

    grader = CRAGRelevanceGrader(threshold_high=0.70, threshold_low=0.50)
    result = grader.grade(query, distractors)

    assert result.decision == GradingDecision.INCORRECT
    assert result.confidence_score < 0.50
    assert result.fallback_action == "trigger_fallback_broad_search"


def test_crag_rewriter(sample_passages):
    relevant_passages, _ = sample_passages
    query = "Who is the mother of the director of film Polish-Russian War?"

    rewriter = CRAGQueryRewriter()
    res = rewriter.rewrite(query, relevant_passages)

    assert res.original_query == query
    assert res.rewritten_query != ""
    assert res.strategy != ""
    assert len(res.extracted_bridge_candidates) > 0
