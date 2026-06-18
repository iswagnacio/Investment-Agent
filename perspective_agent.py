"""
Perspective Analyst Agent Module
================================
The fourth specialist agent. Mirrors TechnicalAnalyzerAgent / FundamentalAnalystAgent /
SentimentAnalystAgent (self-contained class, __init__(model) + analyze(...)), with one
difference: it owns no data calculator. Its inputs are

    (a) an investor framework distilled OFFLINE by distill_investor.py (read from cache), and
    (b) the stock data the lead agent has already gathered.

It applies the investor's framework to that data and returns a structured perspective
signal (score + verdict + narrative) for the lead agent's synthesis + post-blend scoring.

Distillation never happens here. On a cache miss this raises PerspectiveNotCachedError;
the lead agent catches it, prints a hint, and proceeds without perspective.

Usage:
    from perspective_agent import PerspectiveAnalystAgent

    agent = PerspectiveAnalystAgent()
    result = agent.analyze(
        ticker="AAPL",
        investor="Charlie Munger",
        fundamental_data=fund_data,
        technical_data=tech_data,
        sentiment_data=sent_data,
    )
    # result -> {'investor','score','verdict','narrative','metadata', ...}
"""

import os
import re
import json
import glob
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


class PerspectiveNotCachedError(Exception):
    """Raised when a perspective is requested but not present in the cache."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def _normalize_key(investor: str) -> str:
    """Must match distill_investor.normalize_key()."""
    return re.sub(r"[^a-z0-9]+", "-", investor.lower()).strip("-")


# Verdict -> base score, used only as a fallback if the model omits a numeric score.
_VERDICT_BASE = {
    "strong buy": 88,
    "buy": 72,
    "hold": 50,
    "too hard pile": 50,
    "too hard": 50,
    "sell": 28,
    "strong sell": 12,
}


class PerspectiveAnalystAgent:
    """
    Applies an offline-distilled investor framework to stock data.

    Output modes (analysis_type):
      - "for_synthesis" (default): structured dict for the lead agent
      - "comprehensive": same dict, but 'narrative' carries a longer prose write-up
    """

    def __init__(self, model: str = "claude-sonnet-4-5",
                 cache_dir: str = "data/perspectives_cache"):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.cache_dir = cache_dir
        self.llm = ChatAnthropic(model_name=model, temperature=0, max_tokens=4096)

    # =========================================================================
    # Cache reads (the offline distiller is the only writer)
    # =========================================================================
    def load_framework(self, investor: str) -> Dict[str, Any]:
        """
        Load a distilled perspective from cache.

        Returns {'framework': <framework.json>, 'metadata': <metadata.json>, 'key': str}.
        Raises PerspectiveNotCachedError if not found.
        """
        key = _normalize_key(investor)
        base = os.path.join(self.cache_dir, key)
        framework_path = os.path.join(base, "framework.json")
        if not os.path.isfile(framework_path):
            raise PerspectiveNotCachedError(
                f"No cached perspective for '{investor}' (looked in {base}/). "
                f"Run:  python distill_investor.py \"{investor}\""
            )
        with open(framework_path, encoding="utf-8") as fh:
            framework = json.load(fh)

        metadata = {}
        meta_path = os.path.join(base, "metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)

        return {"framework": framework, "metadata": metadata, "key": key}

    def list_available_perspectives(self) -> List[Dict[str, Any]]:
        """Scan the cache and return a summary of every distilled perspective."""
        out: List[Dict[str, Any]] = []
        for meta_path in glob.glob(os.path.join(self.cache_dir, "*", "metadata.json")):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                q = meta.get("quality_metrics", {})
                out.append({
                    "name": meta.get("investor_name"),
                    "key": meta.get("normalized_key"),
                    "confidence": q.get("confidence"),
                    "passes_gate": q.get("passes_gate"),
                    "mental_models": q.get("mental_models_count"),
                    "distillation_date": meta.get("distillation_date"),
                })
            except Exception:
                continue
        return sorted(out, key=lambda x: (x.get("name") or "").lower())

    def validate_perspective_quality(self, investor: str) -> Dict[str, Any]:
        """Report whether a cached perspective clears its own quality gate."""
        loaded = self.load_framework(investor)  # raises if missing
        q = loaded["metadata"].get("quality_metrics", {})
        return {
            "investor": loaded["framework"].get("investor_name", investor),
            "passes_gate": q.get("passes_gate", False),
            "confidence": q.get("confidence", "Low"),
            "mental_models_count": q.get("mental_models_count", 0),
            "decision_heuristics_count": q.get("decision_heuristics_count", 0),
            "recommendation": (
                "High confidence" if q.get("confidence") == "High"
                else "Use with caution" if q.get("passes_gate")
                else "Below gate — re-distill or skip"
            ),
        }

    # =========================================================================
    # Prompt construction
    # =========================================================================
    def _build_system_prompt(self, framework: Dict[str, Any]) -> str:
        name = framework.get("investor_name", "this investor")
        parts = [
            f"You are an investment analyst applying {name}'s framework to a single stock.",
            "Reason strictly within this framework. Do not invent positions the framework "
            "does not support. If the data is insufficient for a dimension, say so.",
            "",
            f"SUMMARY OF {name.upper()}'S APPROACH:",
            framework.get("summary", "(none)"),
        ]

        mm = framework.get("mental_models", []) or []
        if mm:
            parts.append("\nMENTAL MODELS:")
            for m in mm:
                line = f"- {m.get('name', '')}: {m.get('description', '')}"
                if m.get("how_applied"):
                    line += f" (Applied: {m['how_applied']})"
                parts.append(line)

        dh = framework.get("decision_heuristics", []) or []
        if dh:
            parts.append("\nDECISION HEURISTICS:")
            for h in dh:
                ex = f"  e.g. {h['example']}" if h.get("example") else ""
                parts.append(f"- {h.get('rule', '')}{ex}")

        dims = framework.get("analysis_dimensions", []) or []
        if dims:
            parts.append("\nDIMENSIONS THIS INVESTOR EXAMINES: " + ", ".join(dims))

        coc = framework.get("circle_of_competence", {}) or {}
        if coc.get("understands"):
            parts.append("ENGAGES: " + ", ".join(coc["understands"]))
        if coc.get("avoids"):
            parts.append("AVOIDS (likely 'Too Hard Pile'): " + ", ".join(coc["avoids"]))

        rf = framework.get("red_flags", []) or []
        if rf:
            parts.append("RED FLAGS (pass / sell triggers): " + "; ".join(rf))

        parts.append(
            "\nOUTPUT: Respond with ONLY a JSON object, no prose, no markdown fences:\n"
            "{\n"
            '  "verdict": "Strong Buy" | "Buy" | "Hold" | "Sell" | "Strong Sell" | "Too Hard Pile",\n'
            '  "confidence": "High" | "Medium" | "Low",\n'
            '  "score": <integer 0-100, where higher = more attractive from this framework>,\n'
            '  "findings": [ {"dimension": str, "assessment": str} ],\n'
            '  "key_risks": [str],\n'
            '  "rationale": "3-5 sentences in this investor\'s analytical voice"\n'
            "}\n"
            "If the company sits outside this investor's circle of competence, return "
            'verdict "Too Hard Pile" with a neutral score near 50 and explain why.'
        )
        return "\n".join(parts)

    def _build_analysis_request(self, ticker: str,
                                fundamental_data: Optional[Dict],
                                technical_data: Optional[Dict],
                                sentiment_data: Optional[Dict]) -> str:
        def block(label: str, payload: Any) -> str:
            if not payload:
                return f"=== {label} ===\n(not available)"
            text = json.dumps(payload, default=str)[:6000]
            return f"=== {label} ===\n{text}"

        return (
            f"ANALYZING: {ticker}\n"
            f"Apply the framework above to {ticker} using the data below. "
            f"Stay focused on {ticker} only.\n\n"
            f"{block('FUNDAMENTAL DATA', fundamental_data)}\n\n"
            f"{block('TECHNICAL DATA', technical_data)}\n\n"
            f"{block('SENTIMENT DATA', sentiment_data)}\n"
        )

    # =========================================================================
    # Main entry
    # =========================================================================
    def analyze(self, ticker: str, investor: str,
                fundamental_data: Optional[Dict] = None,
                technical_data: Optional[Dict] = None,
                sentiment_data: Optional[Dict] = None,
                analysis_type: str = "for_synthesis") -> Dict[str, Any]:
        """
        Apply `investor`'s distilled framework to `ticker`.

        Raises PerspectiveNotCachedError if the investor isn't in the cache.
        Returns a structured dict (see module docstring).
        """
        ticker = ticker.upper()
        loaded = self.load_framework(investor)  # raises if not cached
        framework = loaded["framework"]
        metadata = loaded["metadata"]
        distill_q = metadata.get("quality_metrics", {})

        system_prompt = self._build_system_prompt(framework)
        user_prompt = self._build_analysis_request(
            ticker, fundamental_data, technical_data, sentiment_data
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        text = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = json.loads(_strip_fences(text))
        except json.JSONDecodeError:
            parsed = {"verdict": "Hold", "confidence": "Low", "score": 50,
                      "findings": [], "key_risks": [],
                      "rationale": "Could not parse structured perspective output."}

        score = self._resolve_score(parsed)
        narrative = self._format_narrative(framework, parsed, analysis_type)

        return {
            "investor": framework.get("investor_name", investor),
            "score": score,
            "verdict": parsed.get("verdict", "Hold"),
            "narrative": narrative,
            "key_risks": parsed.get("key_risks", []),
            "metadata": {
                "investor": framework.get("investor_name", investor),
                "key": loaded["key"],
                "verdict": parsed.get("verdict", "Hold"),
                # confidence of THIS analysis (data-driven)
                "analysis_confidence": parsed.get("confidence", "Low"),
                # confidence of the DISTILLATION (consumed by _effective_perspective_weight)
                "distillation_confidence": distill_q.get("confidence", "Low"),
                "passes_gate": distill_q.get("passes_gate", False),
                "cache_hit": True,
            },
        }

    # =========================================================================
    # Helpers
    # =========================================================================
    @staticmethod
    def _resolve_score(parsed: Dict[str, Any]) -> float:
        raw = parsed.get("score")
        if isinstance(raw, (int, float)):
            return float(max(0, min(100, raw)))
        # fallback: verdict base, nudged by stated confidence
        base = _VERDICT_BASE.get(str(parsed.get("verdict", "hold")).lower(), 50)
        factor = {"High": 1.0, "Medium": 0.9, "Low": 0.8}.get(parsed.get("confidence"), 0.9)
        return round(50 + (base - 50) * factor, 1)

    @staticmethod
    def _format_narrative(framework: Dict[str, Any], parsed: Dict[str, Any],
                          analysis_type: str) -> str:
        name = framework.get("investor_name", "This investor")
        lines = [f"{name}'s read: {parsed.get('rationale', '')}".strip()]
        if analysis_type == "comprehensive":
            for f in parsed.get("findings", []) or []:
                if f.get("dimension") or f.get("assessment"):
                    lines.append(f"- {f.get('dimension','')}: {f.get('assessment','')}")
            if parsed.get("key_risks"):
                lines.append("Key risks: " + "; ".join(parsed["key_risks"]))
        return "\n".join(lines).strip()


# =============================================================================
# CLI (sanity-check a cached perspective against one ticker)
# =============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Apply a cached perspective to a ticker.")
    ap.add_argument("investor")
    ap.add_argument("ticker")
    ap.add_argument("--list", action="store_true", help="List cached perspectives and exit")
    args = ap.parse_args()

    agent = PerspectiveAnalystAgent()

    if args.list:
        for p in agent.list_available_perspectives():
            print(f"  {p['name']:<28} conf={p['confidence']:<7} gate={p['passes_gate']}")
        raise SystemExit(0)

    # minimal smoke test: pull fundamentals via the existing calculator if present
    fund = None
    try:
        from fundamental import FundamentalCalculator
        fund = FundamentalCalculator().get_all_fundamentals(args.ticker.upper())
    except Exception as e:
        print(f"(no fundamental data: {e})")

    result = agent.analyze(args.investor, args.ticker, fundamental_data=fund,
                           analysis_type="comprehensive") if False else \
             agent.analyze(args.ticker, args.investor, fundamental_data=fund,
                           analysis_type="comprehensive")
    print(f"\n{result['investor']} on {args.ticker.upper()}: "
          f"{result['verdict']}  (score {result['score']})")
    print(f"distillation confidence: {result['metadata']['distillation_confidence']}")
    print("\n" + result["narrative"])