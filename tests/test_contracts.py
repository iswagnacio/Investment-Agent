"""Unit tests for AnalysisRecord persistence and report rendering — no network."""

import json

from investment import config
from investment.contracts import AnalysisRecord, SCHEMA_VERSION
from investment.reporting import render_report, render_quick_line


def _record(**overrides) -> AnalysisRecord:
    base = dict(
        ticker="mu",
        company="Micron Technology",
        sector="Technology",
        industry="Semiconductors",
        current_price=123.45,
        model=config.STRONG_MODEL,
        signals={
            "technical": {"score": 62.0, "breakdown": {}},
            "fundamental": {"score": 71.5, "breakdown": {}},
            "sentiment": {"score": 55.0, "breakdown": {}},
            "perspective": {"score": 80.0, "verdict": "Buy", "narrative": "Bottleneck supplier."},
        },
        weights={"technical": 0.298, "fundamental": 0.382, "sentiment": 0.17, "perspective": 0.15},
        overall_score=66.4,
        signal="Buy",
        confidence=70,
        thesis="MU (Micron Technology) shows converging strength across domains.",
        key_risks=["HBM oversupply by 2027"],
        data_flags=["detailed fundamental fetched — conflicting quality signals"],
    )
    base.update(overrides)
    return AnalysisRecord(**base)


class TestAnalysisRecord:
    def test_ticker_uppercased_and_defaults_stamped(self):
        rec = _record()
        assert rec.ticker == "MU"
        assert rec.schema_version == SCHEMA_VERSION
        assert rec.generated_at and rec.run_id

    def test_save_layout_and_roundtrip(self, tmp_path):
        rec = _record()
        path = rec.save(runs_dir=tmp_path)
        # <runs_dir>/<YYYY-MM-DD>/<TICKER>.json
        assert path.name == "MU.json"
        assert path.parent.parent == tmp_path
        assert path.parent.name == rec.generated_at[:10]

        loaded = AnalysisRecord.load(path)
        assert loaded.to_dict() == rec.to_dict()

    def test_same_day_save_overwrites(self, tmp_path):
        _record(overall_score=60.0).save(runs_dir=tmp_path)
        path = _record(overall_score=42.0).save(runs_dir=tmp_path)
        assert AnalysisRecord.load(path).overall_score == 42.0
        day_dir = path.parent
        assert len(list(day_dir.glob("MU.json"))) == 1

    def test_load_ignores_unknown_fields(self, tmp_path):
        # Forward compatibility: a future schema may add fields
        path = _record().save(runs_dir=tmp_path)
        raw = json.loads(path.read_text())
        raw["some_future_field"] = {"x": 1}
        path.write_text(json.dumps(raw))
        assert AnalysisRecord.load(path).ticker == "MU"

    def test_latest_picks_most_recent_day(self, tmp_path):
        old = _record(generated_at="2026-06-01T10:00:00", overall_score=40.0)
        new = _record(generated_at="2026-07-01T10:00:00", overall_score=70.0)
        old.save(runs_dir=tmp_path)
        new.save(runs_dir=tmp_path)

        assert AnalysisRecord.latest("MU", runs_dir=tmp_path).overall_score == 70.0
        # before_day is exclusive -> the Phase 2 "diff against previous run" query
        prev = AnalysisRecord.latest("MU", runs_dir=tmp_path, before_day="2026-07-01")
        assert prev.overall_score == 40.0

    def test_latest_missing_ticker_is_none(self, tmp_path):
        assert AnalysisRecord.latest("XYZ", runs_dir=tmp_path) is None


class TestRendering:
    def test_report_renders_from_record_only(self):
        report = render_report(_record())
        assert "COMPREHENSIVE INVESTMENT ANALYSIS: MU" in report
        assert "Overall Score:" in report and "66.4" in report
        assert "Signal: Buy" in report
        assert "converging strength" in report          # thesis included
        assert "HBM oversupply" in report               # key risks included
        assert "Perspective" in report and "[Buy]" in report

    def test_report_has_no_profit_calculations(self):
        report = render_report(_record())
        for banned in ("Price Target", "Options", "Stop Loss", "Profit"):
            assert banned not in report

    def test_quick_line(self):
        line = render_quick_line(_record())
        assert "MU" in line and "$123.45" in line and "Buy" in line

    def test_render_handles_sparse_record(self):
        rec = AnalysisRecord(ticker="ZZ")  # nothing but a ticker
        report = render_report(rec)
        assert "ZZ" in report and "Signal: Hold" in report


class TestUniverse:
    def test_get_universe_holds_then_watch_deduped(self, tmp_path):
        csv = tmp_path / "holdings.csv"
        csv.write_text(
            "status,ticker,shares,cost_basis,date_acquired,notes\n"
            "hold,AVGO,1,1,2026-01-01,\n"
            "watch,MU,0,0,,\n"
            "watch,avgo,0,0,,dup in different case\n"
        )
        from investment.universe import PortfolioManager
        pm = PortfolioManager(csv_path=str(csv))
        assert pm.get_universe() == ["AVGO", "MU"]

    def test_summary_has_no_cost_math(self, tmp_path):
        csv = tmp_path / "holdings.csv"
        csv.write_text(
            "status,ticker,shares,cost_basis,date_acquired,notes\n"
            "hold,AVGO,2.2,224.97,2026-01-01,\n"
        )
        from investment.universe import PortfolioManager
        pm = PortfolioManager(csv_path=str(csv))
        summary = pm.get_portfolio_summary()
        assert "AVGO" in summary
        assert "$" not in summary and "Cost" not in summary
