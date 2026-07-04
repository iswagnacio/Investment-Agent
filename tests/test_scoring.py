"""Unit tests for the deterministic scoring layer in lead_agent.py.

No API key, no network, no LLM: these exercise ScoreCalculator's pure
functions plus the perspective post-blend math on a bare agent instance
(constructed without __init__ so nothing external is touched).
"""

import pytest

import config
from lead_agent import (
    GeneralAnalystAgent,
    ScoreCalculator,
    Signal,
    extract_json_from_text,
    validate_ticker_in_text,
)


class TestOverallScore:
    def test_uses_config_blend_weights(self):
        w = config.BLEND_WEIGHTS
        expected = round(80 * w["technical"] + 60 * w["fundamental"] + 40 * w["sentiment"], 1)
        assert ScoreCalculator.calculate_overall_score(80, 60, 40) == expected

    def test_equal_inputs_pass_through(self):
        assert ScoreCalculator.calculate_overall_score(50, 50, 50) == 50.0


class TestScoreToSignal:
    @pytest.mark.parametrize("score,signal", [
        (75, Signal.STRONG_BUY), (74.9, Signal.BUY),
        (60, Signal.BUY), (59.9, Signal.HOLD),
        (40, Signal.HOLD), (39.9, Signal.SELL),
        (25, Signal.SELL), (24.9, Signal.STRONG_SELL),
    ])
    def test_boundaries(self, score, signal):
        assert ScoreCalculator.score_to_signal(score) == signal


def _bare_agent() -> GeneralAnalystAgent:
    """Agent instance without __init__ (no API key / LLM / fetchers)."""
    agent = object.__new__(GeneralAnalystAgent)
    agent.scorer = ScoreCalculator()
    return agent


class TestEffectivePerspectiveWeight:
    def test_clamped_to_max(self):
        agent = _bare_agent()
        got = agent._effective_perspective_weight(
            0.9, {"distillation_confidence": "High"})
        assert got == config.MAX_PERSPECTIVE_WEIGHT

    def test_shrunk_when_distillation_weak(self):
        agent = _bare_agent()
        high = agent._effective_perspective_weight(0.15, {"distillation_confidence": "High"})
        low = agent._effective_perspective_weight(0.15, {"distillation_confidence": "Low"})
        assert high == 0.15
        assert low < high

    def test_negative_requested_clamps_to_zero(self):
        agent = _bare_agent()
        assert agent._effective_perspective_weight(-1, {}) == 0.0


class TestCalculateScores:
    def test_empty_data_is_neutral_hold(self):
        scores = _bare_agent().calculate_scores({})
        assert scores["overall"] == 50.0
        assert scores["signal"] == Signal.HOLD
        assert scores["perspective"] is None
        assert scores["weights"]["perspective"] == 0.0

    def test_perspective_post_blend_identity(self):
        # overall = base*(1-pw) + perspective*pw, and the published weights sum to 1
        data = {"perspective": {"score": 90.0,
                                "metadata": {"distillation_confidence": "High"}}}
        scores = _bare_agent().calculate_scores(data, perspective_weight=0.15)
        assert scores["overall"] == round(50.0 * 0.85 + 90.0 * 0.15, 1)
        assert abs(sum(scores["weights"].values()) - 1.0) < 1e-6

    def test_low_confidence_distillation_moves_score_less(self):
        data_hi = {"perspective": {"score": 90.0,
                                   "metadata": {"distillation_confidence": "High"}}}
        data_lo = {"perspective": {"score": 90.0,
                                   "metadata": {"distillation_confidence": "Low"}}}
        agent = _bare_agent()
        hi = agent.calculate_scores(data_hi, perspective_weight=0.15)["overall"]
        lo = agent.calculate_scores(data_lo, perspective_weight=0.15)["overall"]
        assert lo < hi  # weaker distillation → less influence on the blend


class TestValidationHelpers:
    def test_extract_json_plain(self):
        assert extract_json_from_text('{"score": 70}') == {"score": 70}

    def test_extract_json_from_fenced_block(self):
        text = 'Here you go:\n```json\n{"score": 70, "verdict": "Buy"}\n```\nDone.'
        assert extract_json_from_text(text) == {"score": 70, "verdict": "Buy"}

    def test_extract_json_none_when_absent(self):
        assert extract_json_from_text("no json here") is None

    def test_ticker_validation_passes(self):
        text = "AAPL is strong. AAPL momentum improving. Buy AAPL."
        assert validate_ticker_in_text(text, "AAPL", min_mentions=3)

    def test_ticker_validation_fails_on_wrong_ticker(self):
        text = "MSFT is strong. MSFT momentum improving. Buy MSFT."
        assert not validate_ticker_in_text(text, "AAPL", min_mentions=3)
