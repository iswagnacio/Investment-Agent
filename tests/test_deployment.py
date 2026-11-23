"""Verify the deployable snapshot contains only approved source and runs alone."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts.package_demo import SOURCE_FILES, build_bundle


@pytest.fixture
def source_tree(tmp_path):
    root = tmp_path / 'source'
    for name in SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Synthetic source fixture\n')
    (root / '.streamlit/secrets.toml.example').write_text(
        'ANTHROPIC_API_KEY=""\nFINNHUB_API_KEY=""\nNEWS_API_KEY=""\nALPHA_VANTAGE_API_KEY=""\n'
    )
    (root / '.env.example').write_text(
        'ANTHROPIC_API_KEY=\nFINNHUB_API_KEY=\nNEWS_API_KEY=\nALPHA_VANTAGE_API_KEY=\n'
    )
    for name in ('holdings.csv', '.env', '.git/config', '.streamlit/secrets.toml',
                 'data/runs/private.json', '.agents/private.md', 'investment/unapproved.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('Private synthetic fixture; must not ship')
    return root


def test_shareable_bundle_has_local_handoff_and_blank_env_template(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'demo.zip'
    build_bundle(root, output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {'DEMO_HANDOFF.md', '.env.example'} <= names
        guide = archive.read('DEMO_HANDOFF.md').decode()
        assert 'python3.11 -m venv .venv' in guide
        assert 'python -m pip install -r requirements.txt' in guide
        assert 'python -m streamlit run streamlit_app.py' in guide
        assert 'DEPLOYMENT.md' in guide
        assert 'ANTHROPIC_API_KEY=' in archive.read('.env.example').decode()


def test_nonblank_env_example_is_not_exported(source_tree, tmp_path):
    (source_tree / '.env.example').write_text('ANTHROPIC_API_KEY=private-synthetic-value\n')
    output = tmp_path / 'demo.zip'
    with pytest.raises(ValueError, match='secret'):
        build_bundle(source_tree, output)
    assert not output.exists()


def test_bundle_excludes_private_and_unapproved_files_and_checksums_every_source(source_tree, tmp_path):
    output = tmp_path / 'demo.zip'
    build_bundle(source_tree, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(SOURCE_FILES) | {'DEPLOYMENT_MANIFEST.json'}
        manifest = json.loads(archive.read('DEPLOYMENT_MANIFEST.json'))
        assert manifest['entrypoint'] == 'streamlit_app.py'
        assert manifest['python_version'] == '3.11'
        for name, digest in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    second = tmp_path / 'demo-again.zip'
    build_bundle(source_tree, second)
    assert output.read_bytes() == second.read_bytes()


def test_missing_source_fails_without_writing_archive(source_tree, tmp_path):
    (source_tree / 'streamlit_app.py').unlink()
    output = tmp_path / 'demo.zip'
    with pytest.raises(ValueError, match='streamlit_app.py'):
        build_bundle(source_tree, output)
    assert not output.exists()


def test_symlink_to_private_data_is_rejected(source_tree, tmp_path):
    source = source_tree / 'investment/config.py'
    source.unlink()
    source.symlink_to(source_tree / '.env')
    output = tmp_path / 'demo.zip'
    with pytest.raises(ValueError, match='symlink'):
        build_bundle(source_tree, output)
    assert not output.exists()


def test_credentials_in_selected_source_or_template_abort_export(source_tree, tmp_path):
    output = tmp_path / 'demo.zip'
    path = source_tree / '.streamlit/secrets.toml.example'
    path.write_text('ANTHROPIC_API_KEY="not-empty"\n')
    with pytest.raises(ValueError, match='secret'):
        build_bundle(source_tree, output)
    path.write_text('ANTHROPIC_API_KEY=""\nFINNHUB_API_KEY=""\nNEWS_API_KEY=""\nALPHA_VANTAGE_API_KEY=""\n')
    (source_tree / 'streamlit_app.py').write_text('key = "sk-ant-' + 'x' * 30 + '"\n')
    with pytest.raises(ValueError, match='credential'):
        build_bundle(source_tree, output)
    assert not output.exists()


@pytest.mark.parametrize('name,key', [
    ('investment/config.py', 'FINNHUB_API_KEY'),
    ('investment/config.py', 'NEWS_API_KEY'),
    ('investment/config.py', 'ALPHA_VANTAGE_API_KEY'),
    ('DEPLOYMENT.md', 'ANTHROPIC_API_KEY'),
])
def test_literal_provider_credentials_abort_export(source_tree, tmp_path, name, key):
    (source_tree / name).write_text(f'{key} = "private-synthetic-value"\n')
    output = tmp_path / 'demo.zip'
    with pytest.raises(ValueError, match='credential'):
        build_bundle(source_tree, output)
    assert not output.exists()


def test_extracted_real_app_runs_both_workflows_and_memory_without_private_files(tmp_path):
    root = Path(__file__).resolve().parents[1]
    archive = tmp_path / 'real-demo.zip'
    build_bundle(root, archive)
    exported = tmp_path / 'extracted'
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(exported)
    for name in ('.env', 'holdings.csv', '.git', 'data', '.streamlit/secrets.toml'):
        assert not (exported / name).exists()

    code = r'''
import os, socket
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from langchain_community.llms.fake import FakeListLLM
from langchain_core.callbacks import BaseCallbackHandler
import investment
from investment import web
from investment.contracts import AnalysisRecord
from investment.reporting import render_report
import investment.agents.portfolio as portfolio

assert Path(investment.__file__).resolve().is_relative_to(Path.cwd())
prompts = []
class Capture(BaseCallbackHandler):
    def on_llm_start(self, serialized, batch, **kwargs):
        prompts.extend(batch)
llm = FakeListLLM(responses=[
    "Thought: Read holdings.\nAction: get_holdings\nAction Input: unused",
    "Thought: Ready.\nFinal Answer: AAPL has 10 shares; MSFT has 5 shares.",
    "Thought: Ready.\nFinal Answer: AAPL is largest by share count, with 10 shares.",
], callbacks=[Capture()])
report = render_report(AnalysisRecord(
    ticker="AAPL", signals={"technical":{"score":61},"fundamental":{"score":72},"sentiment":{"score":53}},
    overall_score=66, confidence=70, thesis="Synthetic offline research smoke test.",
))
with patch.object(socket.socket, "connect", side_effect=AssertionError("No network in smoke test")), \
     patch.object(web, "run_research", return_value=report), \
     patch.object(portfolio, "ChatAnthropic", return_value=llm):
    at = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
    assert not at.exception
    at.text_input(key="ticker").set_value("AAPL")
    at.button(key="research_submit").click().run()
    assert not at.exception and len(at.metric) == 6
    at.radio(key="workflow").set_value("Portfolio Assistant").run()
    at.text_area(key="portfolio_question").set_value("What's in my portfolio?")
    at.button(key="portfolio_submit").click().run()
    assert not at.exception and len(at.session_state["portfolio_history"]) == 1
    assert any("get_holdings" in item.value for item in at.text)
    at.text_area(key="portfolio_question").set_value("Tell me about my largest holding")
    at.button(key="portfolio_submit").click().run()
    assert not at.exception and len(at.session_state["portfolio_history"]) == 2
    assert "AAPL has 10 shares" in prompts[-1]
    at.button(key="reset_portfolio").click().run()
    assert not at.exception and at.session_state["portfolio_history"] == []
assert not Path("holdings.csv").exists()
assert not Path("data/runs").exists()
print("Extracted snapshot: both workflows, real read tool, prior context, and reset passed offline.")
'''
    environment = {name: value for name, value in os.environ.items()
                   if not any(part in name.upper() for part in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
    environment['ANTHROPIC_API_KEY'] = 'offline-placeholder-not-a-credential'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=exported, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Extracted snapshot:' in result.stdout
