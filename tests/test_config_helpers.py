"""Unit tests for the shared helpers in config.py — pure functions, no network."""

import config


class TestNormalizeKey:
    def test_name_with_spaces(self):
        assert config.normalize_key("Charlie Munger") == "charlie-munger"

    def test_handle_with_at(self):
        assert config.normalize_key("Serenity@aleabitoreddit") == "serenity-aleabitoreddit"

    def test_strips_leading_trailing_junk(self):
        assert config.normalize_key("  @Buffett!  ") == "buffett"

    def test_idempotent(self):
        once = config.normalize_key("Warren E. Buffett")
        assert config.normalize_key(once) == once


class TestStripFences:
    def test_plain_json_untouched(self):
        assert config.strip_fences('{"a": 1}') == '{"a": 1}'

    def test_json_fence(self):
        assert config.strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_bare_fence(self):
        assert config.strip_fences('```\n{"a": 1}\n```') == '{"a": 1}'


class TestConfigInvariants:
    def test_blend_weights_sum_to_one(self):
        assert abs(sum(config.BLEND_WEIGHTS.values()) - 1.0) < 1e-9

    def test_perspective_weight_bounds(self):
        assert 0 < config.DEFAULT_PERSPECTIVE_WEIGHT <= config.MAX_PERSPECTIVE_WEIGHT < 1

    def test_paths_anchored_to_repo_root(self):
        # Every configured path must be absolute (cwd-independent) and under ROOT.
        for p in (config.DATA_DIR, config.PERSPECTIVES_CACHE_DIR, config.RUNS_DIR,
                  config.WORKSPACES_DIR, config.HOLDINGS_CSV):
            assert p.is_absolute()
            assert config.ROOT in p.parents
