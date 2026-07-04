"""Unit tests for the assessment registry: parsing, trust-gated weighting,
post-blend capping, workspaces, and the lead agent's custom blend. Offline."""

import json

import pytest

from investment import config
import investment.registry as registry_mod
from investment.contracts import AnalysisRecord
from investment.registry import (
    MATURITY_FLOOR,
    MATURITY_RUNS,
    MAX_TOTAL_POST_WEIGHT,
    CustomAgentSpec,
    Permissions,
    Registry,
    appearances,
    cap_post_blend,
    effective_custom_weight,
    ensure_workspace,
    list_workspaces,
    load_registry,
)


class TestLoadRegistry:
    def test_missing_file_gives_defaults(self, tmp_path):
        reg = load_registry(tmp_path / "nope.toml")
        assert reg.core_weights == config.BLEND_WEIGHTS
        assert reg.custom == {}

    def test_full_parse(self, tmp_path):
        p = tmp_path / "registry.toml"
        p.write_text(
            "[core]\ntechnical = 0.3\nfundamental = 0.5\nsentiment = 0.2\n"
            "[agents.bottleneck-screen]\nenabled = true\nweight = 0.12\nmax_weight = 0.1\n"
            "[agents.bottleneck-screen.permissions]\n"
            'data_sources = ["fundamental"]\nmax_llm_calls = 1\n'
        )
        reg = load_registry(p)
        assert reg.core_weights == {"technical": 0.3, "fundamental": 0.5, "sentiment": 0.2}
        spec = reg.custom["bottleneck-screen"]
        assert spec.weight == 0.12 and spec.max_weight == 0.1
        assert spec.permissions.data_sources == ["fundamental"]
        assert reg.enabled_custom() == [spec]

    def test_bad_core_sum_raises(self, tmp_path):
        p = tmp_path / "registry.toml"
        p.write_text("[core]\ntechnical = 0.5\nfundamental = 0.5\nsentiment = 0.2\n")
        with pytest.raises(ValueError, match="sum to 1.0"):
            load_registry(p)

    def test_unknown_data_source_raises(self, tmp_path):
        p = tmp_path / "registry.toml"
        p.write_text('[agents.x]\n[agents.x.permissions]\ndata_sources = ["insider-trades"]\n')
        with pytest.raises(ValueError, match="unknown data_sources"):
            load_registry(p)

    def test_disabled_agent_excluded_from_enabled(self, tmp_path):
        p = tmp_path / "registry.toml"
        p.write_text("[agents.x]\nenabled = false\n")
        reg = load_registry(p)
        assert "x" in reg.custom and reg.enabled_custom() == []


class TestTrustGatedWeight:
    def _spec(self, weight=0.10, max_weight=0.10):
        return CustomAgentSpec(key="x", weight=weight, max_weight=max_weight)

    def test_mature_high_confidence_gets_full_weight(self):
        w = effective_custom_weight(self._spec(), "High", n_appearances=MATURITY_RUNS)
        assert w == 0.10

    def test_new_agent_starts_at_floor(self):
        w = effective_custom_weight(self._spec(), "High", n_appearances=0)
        assert w == round(0.10 * MATURITY_FLOOR, 3)

    def test_maturity_ramps(self):
        half = effective_custom_weight(self._spec(), "High", n_appearances=MATURITY_RUNS // 2)
        full = effective_custom_weight(self._spec(), "High", n_appearances=MATURITY_RUNS)
        assert half == round(0.10 * 0.5, 3) and half < full

    def test_low_confidence_shrinks(self):
        hi = effective_custom_weight(self._spec(), "High", n_appearances=MATURITY_RUNS)
        lo = effective_custom_weight(self._spec(), "Low", n_appearances=MATURITY_RUNS)
        assert lo == round(hi * 0.4, 3)

    def test_capped_by_max_weight_and_global_cap(self):
        w = effective_custom_weight(self._spec(weight=0.9, max_weight=0.5),
                                    "High", n_appearances=MATURITY_RUNS)
        assert w == config.MAX_PERSPECTIVE_WEIGHT


class TestAppearances:
    def test_counts_records_containing_agent_signal(self, tmp_path):
        AnalysisRecord(ticker="MU", generated_at="2026-07-01T10:00:00",
                       signals={"x": {"score": 60}}).save(runs_dir=tmp_path)
        AnalysisRecord(ticker="KO", generated_at="2026-07-02T10:00:00",
                       signals={"technical": {"score": 50}}).save(runs_dir=tmp_path)
        assert appearances("x", runs_dir=tmp_path) == 1
        assert appearances("technical", runs_dir=tmp_path) == 1
        assert appearances("nope", runs_dir=tmp_path) == 0

    def test_empty_dir(self, tmp_path):
        assert appearances("x", runs_dir=tmp_path / "none") == 0


class TestCapPostBlend:
    def test_under_cap_untouched(self):
        entries = [("perspective", 80, 0.15), ("x", 60, 0.10)]
        assert cap_post_blend(entries) == entries

    def test_over_cap_scaled_proportionally(self):
        entries = [("a", 80, 0.30), ("b", 60, 0.30)]
        capped = cap_post_blend(entries)
        total = sum(w for _, _, w in capped)
        assert total == pytest.approx(MAX_TOTAL_POST_WEIGHT, abs=1e-6)
        # proportions preserved
        assert capped[0][2] == capped[1][2]


class TestWorkspaces:
    def test_ensure_and_list(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "WORKSPACES_DIR", tmp_path)
        ws = ensure_workspace("Bottleneck Screen!")
        assert ws == tmp_path / "bottleneck-screen"
        assert ws.is_dir()
        assert list_workspaces() == []  # no framework.json yet
        (ws / "framework.json").write_text("{}")
        assert list_workspaces() == ["bottleneck-screen"]


class TestLeadAgentCustomBlend:
    def _agent(self, reg):
        from investment.agents.lead import GeneralAnalystAgent
        from investment.scoring import ScoreCalculator
        agent = object.__new__(GeneralAnalystAgent)
        agent.scorer = ScoreCalculator()
        agent.registry = reg
        return agent

    def _registry_with(self, key="x", **spec_kw):
        reg = Registry()
        reg.custom[key] = CustomAgentSpec(key=key, **spec_kw)
        return reg

    def test_custom_assessment_folds_into_blend(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)  # hermetic maturity count
        reg = self._registry_with(weight=0.10, max_weight=0.10)
        agent = self._agent(reg)
        data = {"custom_assessments": {"x": {
            "score": 90.0,
            "metadata": {"distillation_confidence": "High"},
        }}}
        # n_appearances comes from the runs dir; brand-new agent -> floor
        scores = agent.calculate_scores(data)
        w = scores["weights"]["x"]
        assert w == round(0.10 * MATURITY_FLOOR, 3)
        assert scores["overall"] == round(50.0 * (1 - w) + 90.0 * w, 1)
        assert sum(scores["weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert scores["custom"]["x"]["score"] == 90.0

    def test_unregistered_assessment_ignored(self):
        agent = self._agent(Registry())
        data = {"custom_assessments": {"ghost": {"score": 99.0, "metadata": {}}}}
        scores = agent.calculate_scores(data)
        assert "ghost" not in scores["weights"]
        assert scores["overall"] == 50.0

    def test_perspective_and_custom_together(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        reg = self._registry_with(weight=0.10)
        agent = self._agent(reg)
        data = {
            "perspective": {"score": 80.0,
                            "metadata": {"distillation_confidence": "High"}},
            "custom_assessments": {"x": {
                "score": 90.0, "metadata": {"distillation_confidence": "High"}}},
        }
        scores = agent.calculate_scores(data, perspective_weight=0.15)
        w_p = scores["weights"]["perspective"]
        w_x = scores["weights"]["x"]
        assert w_p == 0.15 and w_x > 0
        expected = 50.0 * (1 - w_p - w_x) + 80.0 * w_p + 90.0 * w_x
        assert scores["overall"] == round(expected, 1)
        assert sum(scores["weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_registry_core_weights_replace_config(self):
        default_agent = self._agent(Registry())
        assert default_agent.calculate_scores({})["weights"]["technical"] == \
            config.BLEND_WEIGHTS["technical"]

        reg = Registry(core_weights={"technical": 0.6, "fundamental": 0.3, "sentiment": 0.1})
        assert self._agent(reg).calculate_scores({})["weights"]["technical"] == 0.6
