"""Build a history-free, allowlisted source ZIP for the public web demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    ".env.example",
    ".gitignore",
    ".streamlit/config.toml",
    ".streamlit/secrets.toml.example",
    "DEMO_HANDOFF.md",
    "DEPLOYMENT.md",
    "pyproject.toml",
    "registry.toml",
    "requirements.txt",
    "streamlit_app.py",
    "investment/__init__.py",
    "investment/config.py",
    "investment/contracts.py",
    "investment/conversation.py",
    "investment/registry.py",
    "investment/reporting.py",
    "investment/safety.py",
    "investment/scoring.py",
    "investment/universe.py",
    "investment/web.py",
    "investment/agents/__init__.py",
    "investment/agents/fundamental.py",
    "investment/agents/lead.py",
    "investment/agents/perspective.py",
    "investment/agents/portfolio.py",
    "investment/agents/sentiment.py",
    "investment/agents/technical.py",
    "investment/calculators/__init__.py",
    "investment/calculators/fundamental.py",
    "investment/calculators/sentiment.py",
    "investment/calculators/technical.py",
)
SECRET_KEYS = frozenset((
    "ANTHROPIC_API_KEY", "FINNHUB_API_KEY", "NEWS_API_KEY", "ALPHA_VANTAGE_API_KEY",
))
ENV_TEMPLATE_KEYS = SECRET_KEYS | {"TAVILY_API_KEY", "X_BEARER_TOKEN"}
_CREDENTIAL_PATTERNS = (
    re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    # Any literal value assigned to a provider-style secret name is suspect,
    # even when the provider has no recognizable token prefix.
    re.compile(
        rb"(?mi)^[ \t]*(?:export[ \t]+)?[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)"
        rb"[ \t]*[:=][ \t]*([\"'])[^\"'\r\n]+\1"
    ),
)


def _check_template(data: bytes) -> None:
    try:
        values = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ValueError("Invalid blank secret template") from exc
    if set(values) != SECRET_KEYS or any(value != "" for value in values.values()):
        raise ValueError("secret template must contain only four blank top-level keys")


def _check_env_template(data: bytes) -> None:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("Invalid blank secret environment template") from exc
    found = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in ENV_TEMPLATE_KEYS or value.strip():
            raise ValueError("secret environment template must contain only blank approved keys")
        found.add(key)
    if not SECRET_KEYS <= found:
        raise ValueError("secret environment template is missing required blank keys")


def build_bundle(root: Path, output: Path) -> Path:
    """Validate and export approved files; never inspect local secrets or holdings."""
    root = Path(root).resolve()
    output = Path(output)
    sources: dict[str, bytes] = {}
    for name in SOURCE_FILES:
        candidate = root / name
        if any(part.is_symlink() for part in (candidate, *candidate.parents) if part != root):
            raise ValueError(f"Source path contains symlink: {name}")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise ValueError(f"Missing or invalid source file: {name}")
        data = candidate.read_bytes()
        if name == ".streamlit/secrets.toml.example":
            _check_template(data)
        if name == ".env.example":
            _check_env_template(data)
        if any(pattern.search(data) for pattern in _CREDENTIAL_PATTERNS):
            raise ValueError(f"credential-like content in approved source: {name}")
        sources[name] = data

    manifest = {
        "schema_version": 1,
        "entrypoint": "streamlit_app.py",
        "python_version": "3.11",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(sources.items())},
    }
    sources["DEPLOYMENT_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(sources.items()):
                entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/Investment-Agent-500D-demo.zip")
    args = parser.parse_args()
    print(build_bundle(ROOT, args.output))
