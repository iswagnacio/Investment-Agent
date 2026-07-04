"""Unit tests for perspective plumbing: distiller quality gate + agent score
resolution + the cache-key contract between the two. No network, no LLM."""

import pytest

from investment import config
from investment.offline.distill_investor import compute_quality
from investment.agents.perspective import PerspectiveAnalystAgent, _normalize_key


class TestQualityGate:
    def _framework(self, mm=3, dh=3, overall="strong"):
        return {
            "mental_models": [{"name": f"m{i}"} for i in range(mm)],
            "decision_heuristics": [{"rule": f"r{i}"} for i in range(dh)],
            "evidence": {"overall": overall},
        }

    def test_passes_with_enough_methodology(self):
        q = compute_quality(self._framework())
        assert q["passes_gate"] and q["confidence"] == "High"

    def test_fails_on_thin_mental_models(self):
        assert not compute_quality(self._framework(mm=2))["passes_gate"]

    def test_fails_on_insufficient_evidence(self):
        q = compute_quality(self._framework(overall="insufficient"))
        assert not q["passes_gate"] and q["confidence"] == "Low"

    def test_weak_evidence_is_medium_confidence(self):
        assert compute_quality(self._framework(overall="weak"))["confidence"] == "Medium"

    def test_empty_framework_fails_closed(self):
        q = compute_quality({})
        assert not q["passes_gate"] and q["confidence"] == "Low"


class TestResolveScore:
    def test_numeric_score_clamped(self):
        assert PerspectiveAnalystAgent._resolve_score({"score": 150}) == 100.0
        assert PerspectiveAnalystAgent._resolve_score({"score": -5}) == 0.0

    def test_numeric_score_passthrough(self):
        assert PerspectiveAnalystAgent._resolve_score({"score": 72}) == 72.0

    def test_fallback_from_verdict(self):
        got = PerspectiveAnalystAgent._resolve_score(
            {"verdict": "Strong Buy", "confidence": "High"})
        assert got > 50

    def test_fallback_confidence_shrinks_toward_neutral(self):
        hi = PerspectiveAnalystAgent._resolve_score({"verdict": "Buy", "confidence": "High"})
        lo = PerspectiveAnalystAgent._resolve_score({"verdict": "Buy", "confidence": "Low"})
        assert 50 < lo < hi

    def test_unknown_verdict_is_neutral(self):
        assert PerspectiveAnalystAgent._resolve_score({"verdict": "???"}) == 50.0


class TestCacheKeyContract:
    def test_agent_and_distiller_share_one_normalize_key(self):
        # Phase 0 collapsed the duplicated implementations into config.
        from investment.offline.distill_investor import normalize_key as distiller_key
        assert distiller_key is config.normalize_key
        assert _normalize_key is config.normalize_key

    def test_distiller_cache_dir_matches_agent_default(self):
        import investment.offline.distill_investor as distill_investor
        agent_default = PerspectiveAnalystAgent.__init__.__defaults__[1]
        assert distill_investor.CACHE_DIR == agent_default == str(config.PERSPECTIVES_CACHE_DIR)
