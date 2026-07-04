"""Unit tests for the headless runner: config, universe resolution, and the
diff digest. No network, no LLM — evaluation itself is exercised offline."""

import config
import runner
from contracts import AnalysisRecord
from runner import (
    TickerOutcome,
    classify_outcome,
    load_runner_config,
    render_digest,
    resolve_universe,
)


def _rec(ticker="MU", signal="Buy", score=66.0, confidence=70.0):
    return AnalysisRecord(ticker=ticker, signal=signal,
                          overall_score=score, confidence=confidence)


class TestRunnerConfig:
    def test_missing_file_gives_defaults(self, tmp_path):
        cfg = load_runner_config(tmp_path / "nope.toml")
        assert cfg["tickers"] == []
        assert cfg["perspective"] is None
        assert cfg["perspective_weight"] == config.DEFAULT_PERSPECTIVE_WEIGHT
        assert cfg["parallel"] == 1

    def test_file_parsed_and_normalized(self, tmp_path):
        p = tmp_path / "runner.toml"
        p.write_text(
            '[universe]\ntickers = ["mu", "avgo"]\n'
            '[analysis]\nperspective = "serenity-aleabitoreddit"\n'
            'perspective_weight = 0.2\n'
            '[run]\nparallel = 3\n'
        )
        cfg = load_runner_config(p)
        assert cfg["tickers"] == ["MU", "AVGO"]
        assert cfg["perspective"] == "serenity-aleabitoreddit"
        assert cfg["perspective_weight"] == 0.2
        assert cfg["parallel"] == 3

    def test_empty_perspective_string_means_none(self, tmp_path):
        p = tmp_path / "runner.toml"
        p.write_text('[analysis]\nperspective = ""\n')
        assert load_runner_config(p)["perspective"] is None


class TestResolveUniverse:
    def test_cli_beats_config(self):
        cfg = {"tickers": ["KO"]}
        assert resolve_universe(cfg, ["mu"]) == ["MU"]

    def test_config_beats_holdings(self):
        assert resolve_universe({"tickers": ["KO"]}) == ["KO"]

    def test_falls_back_to_holdings(self, monkeypatch, tmp_path):
        csv = tmp_path / "holdings.csv"
        csv.write_text(
            "status,ticker,shares,cost_basis,date_acquired,notes\n"
            "hold,AVGO,1,1,2026-01-01,\nwatch,MU,0,0,,\n"
        )
        import portfolio
        orig = portfolio.PortfolioManager
        monkeypatch.setattr(portfolio, "PortfolioManager",
                            lambda: orig(csv_path=str(csv)))
        assert resolve_universe({"tickers": []}) == ["AVGO", "MU"]


class TestClassify:
    def test_failed(self):
        assert classify_outcome(TickerOutcome("MU", error="boom")) == "failed"
        assert classify_outcome(TickerOutcome("MU")) == "failed"

    def test_new(self):
        assert classify_outcome(TickerOutcome("MU", record=_rec())) == "new"

    def test_signal_change(self):
        o = TickerOutcome("MU", record=_rec(signal="Hold", score=55),
                          previous=_rec(signal="Buy", score=62))
        assert classify_outcome(o) == "signal_change"

    def test_notable_move_without_flip(self):
        o = TickerOutcome("MU", record=_rec(score=72), previous=_rec(score=61))
        assert classify_outcome(o) == "notable_move"

    def test_unchanged(self):
        o = TickerOutcome("MU", record=_rec(score=66), previous=_rec(score=64))
        assert classify_outcome(o) == "unchanged"


class TestDigest:
    def test_digest_sections(self):
        outcomes = [
            TickerOutcome("MU", record=_rec("MU", "Hold", 52, 55),
                          previous=_rec("MU", "Buy", 66, 70)),
            TickerOutcome("KO", record=_rec("KO", "Hold", 44),
                          previous=_rec("KO", "Hold", 55)),
            TickerOutcome("AVGO", record=_rec("AVGO", "Buy", 68)),
            TickerOutcome("NOK", error="fetch failed"),
            TickerOutcome("RF", record=_rec("RF", "Hold", 50),
                          previous=_rec("RF", "Hold", 51)),
        ]
        digest = render_digest(outcomes, run_id="test-run", day="2026-07-03")
        assert "**MU: Buy → Hold**" in digest
        assert "KO: 55.0 → 44.0" in digest
        assert "AVGO: Buy (score 68.0)" in digest
        assert "NOK: fetch failed" in digest
        assert "RF (Hold 50.0)" in digest
        assert "4/5 tickers evaluated" in digest

    def test_quiet_day_says_so(self):
        outcomes = [TickerOutcome("RF", record=_rec("RF", "Hold", 50),
                                  previous=_rec("RF", "Hold", 50))]
        digest = render_digest(outcomes, run_id="r", day="d")
        assert "No signal changes" in digest


class TestMain:
    def test_dry_run(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        rc = runner.main(["--dry-run", "--tickers", "MU", "AVGO"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "MU, AVGO" in out
        assert (tmp_path / "").exists()

    def test_empty_universe_exit_2(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
        import portfolio

        class EmptyPM:
            def get_universe(self):
                return []

        monkeypatch.setattr(portfolio, "PortfolioManager", EmptyPM)
        rc = runner.main(["--config", str(tmp_path / "none.toml")])
        assert rc == 2
