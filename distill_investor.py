"""
Offline Investor Perspective Distiller
======================================
Run ONCE per investor, ahead of time. Writes a vetted perspective to the cache
that the analysis pipeline reads at runtime.

This is deliberately NOT wired into lead_agent. Distillation is the slow,
search-dependent, occasionally-messy step, so it lives in its own offline tool.
The runtime only ever READS the cache it produces — fast, deterministic, and
with no search/API dependency at analyze time.

Pipeline (two LLM calls + a source):
    retrieve()         -> fixed templated web searches  -> cheap cleanup pass -> source brief
    load_posts_brief() -> pre-fetched X-posts JSON file  -> cheap cleanup pass -> source brief
    extract()          -> strong model turns the brief into a structured, evidence-gated framework
    save()             -> framework.json (machine-consumed) + SKILL.md (human-vetted) + metadata.json

Two source modes share the same downstream pipeline:
  * Web mode (default): keyword search via Tavily. Good for investors whose thinking
    lives in letters, interviews, and articles.
  * X-posts mode (--posts): distill straight from a file of the investor's own posts.
    Good for X-native investors that general web search under-serves.

Because WE control the output format, the runtime consumes framework.json directly.
No fragile parsing of someone else's SKILL.md layout.

Usage:
    python distill_investor.py "Charlie Munger"
    python distill_investor.py "Warren Buffett" --review            # print SKILL.md after, for vetting
    python distill_investor.py --posts serenity_posts.json          # name/handle derived from the file
    python distill_investor.py "Serenity" --posts serenity_posts.json --key serenity-aleabitoreddit

Env:
    ANTHROPIC_API_KEY   (required)
    TAVILY_API_KEY      (required for web mode; not needed for --posts mode)
"""

import os
import re
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List

import requests
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

CACHE_DIR = "data/perspectives_cache"
STRONG_MODEL = "claude-sonnet-4-5"   # extraction — matches repo convention
CHEAP_MODEL = "claude-haiku-4-5"     # search-result / post cleanup


# =============================================================================
# The ONE provider-specific seam
# =============================================================================
def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Return [{'title', 'url', 'content'}, ...].

    This is the only provider-specific code in the file. The body below targets
    Tavily; to use Brave / SerpAPI / etc., swap the request and the response mapping.

    NOTE: verify the exact request/response shape against your provider's current
    docs — this is the single spot that depends on an external API contract.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set TAVILY_API_KEY, or replace web_search() with your own provider."
        )
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in resp.json().get("results", [])
    ]


# =============================================================================
# Step 1a — Retrieve (web): fixed templated searches + a cheap cleanup pass
# =============================================================================
SEARCH_TEMPLATES = [
    "{investor} investment philosophy",
    "{investor} investment strategy framework",
    "{investor} portfolio holdings positions",
    "{investor} on valuation risk mistakes",
    "{investor} interview letter writing",
]

CLEANUP_SYSTEM = (
    "You are condensing raw web-search results about an investor into a clean brief. "
    "Keep ONLY material about their INVESTMENT thinking, decisions, and track record. "
    "Drop ads, navigation text, duplicates, and unrelated biographical trivia. "
    "Preserve concrete specifics: named holdings, stated rules, direct quotes, dates. "
    "If the material is thin or mostly irrelevant, say so plainly. Never invent anything."
)


def retrieve(investor: str) -> str:
    """Run the fixed searches, then condense into a bounded source brief."""
    raw_chunks: List[str] = []
    for template in SEARCH_TEMPLATES:
        query = template.format(investor=investor)
        try:
            for r in web_search(query):
                if r["content"]:
                    raw_chunks.append(f"[{r['title']}] ({r['url']})\n{r['content']}")
        except Exception as e:  # one failed query shouldn't kill the run
            print(f"  ! search failed for '{query}': {e}")

    if not raw_chunks:
        return ""

    blob = "\n\n---\n\n".join(raw_chunks)
    cheap = ChatAnthropic(model_name=CHEAP_MODEL, temperature=0, max_tokens=4096)
    msg = cheap.invoke(
        [
            SystemMessage(content=CLEANUP_SYSTEM),
            # cap input so the (paid) extraction call stays bounded
            HumanMessage(content=f"INVESTOR: {investor}\n\nRAW RESULTS:\n{blob[:40000]}"),
        ]
    )
    return msg.content if isinstance(msg.content, str) else str(msg.content)


# =============================================================================
# Step 1b — Retrieve (X posts): pre-fetched JSON file + a cheap cleanup pass
# -----------------------------------------------------------------------------
# Same contract as retrieve(): in -> source brief, out -> bounded source brief.
# Drops straight into the pipeline; extract()/compute_quality()/save() unchanged.
# =============================================================================
POST_CHAR_BUDGET = 38000   # packed posts cap before the cleanup pass condenses further
MIN_POST_CHARS = 15        # drop empty / link-only / one-word posts

POSTS_CLEANUP_SYSTEM = (
    "You are condensing one investor's own X/Twitter posts into a clean brief about "
    "their INVESTMENT thinking. Keep only posts that reveal how they analyze companies, "
    "value businesses, size positions, manage risk, or react to being right or wrong. "
    "Preserve concrete specifics: named tickers/holdings, stated rules, numbers, dates, "
    "and direct opinions in their own words. Drop memes, banter, engagement-bait, polls, "
    "off-topic chatter, and pure retweets with no added commentary. When a conviction "
    "shows up repeatedly, note that it recurs rather than collapsing it to a one-off. "
    "If the material is thin on actual methodology, say so plainly. Never invent anything."
)


def _clean_post_text(post: Dict) -> str:
    """Prefer the full long-form note over the truncated `text`, then strip t.co noise."""
    note = (post.get("note_tweet") or {}).get("text")
    text = note or post.get("text", "") or ""
    text = re.sub(r"https?://t\.co/\S+", "", text)          # drop t.co shortlinks
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)            # tidy line breaks
    return text.strip()


def _post_signal(post: Dict) -> int:
    """Cheap salience proxy: what the audience liked + saved."""
    m = post.get("public_metrics", {}) or {}
    return int(m.get("like_count", 0)) + int(m.get("bookmark_count", 0))


def load_posts_brief(path: str) -> str:
    """
    Build a source brief from a pre-fetched X-posts JSON file (the shape the fetcher
    emits: {username, display_name, posts: [{text, note_tweet, public_metrics,
    created_at, ...}]}).

    Posts are packed highest-engagement-first, so if we hit the char budget we drop
    the low-signal tail rather than the posts people actually valued. The cleanup pass
    then condenses the selection into an investment-focused brief — which is also where
    memes / banter / polls get filtered out, keeping extract() honest.
    """
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    posts = payload.get("posts", []) or []
    if not posts:
        return ""

    # highest-engagement first; only the audience-valued tail survives a budget cut
    posts = sorted(posts, key=_post_signal, reverse=True)

    chunks: List[str] = []
    used = 0
    for post in posts:
        text = _clean_post_text(post)
        if len(text) < MIN_POST_CHARS:        # skip image-only / link-only / fragments
            continue
        # Metrics drove the ranking above, but are kept OUT of the model's view: they
        # cost tokens and risk conflating "popular" with "core to the methodology".
        # Date stays — cheap, and lets the model see if a view is recent, stale, or evolving.
        date = (post.get("created_at") or "")[:10]
        chunk = f"[{date}]\n{text}"
        if used + len(chunk) > POST_CHAR_BUDGET:
            break
        chunks.append(chunk)
        used += len(chunk)

    if not chunks:
        return ""

    blob = "\n\n---\n\n".join(chunks)
    handle = payload.get("username", "")
    name = payload.get("display_name", "") or handle
    cheap = ChatAnthropic(model_name=CHEAP_MODEL, temperature=0, max_tokens=4096)
    msg = cheap.invoke(
        [
            SystemMessage(content=POSTS_CLEANUP_SYSTEM),
            HumanMessage(
                content=f"INVESTOR: {name} (@{handle})\n\nTHEIR POSTS:\n{blob[:40000]}"
            ),
        ]
    )
    return msg.content if isinstance(msg.content, str) else str(msg.content)


# =============================================================================
# Step 2 — Extract: structured, evidence-gated investment framework
# =============================================================================
EXTRACTION_SYSTEM = """You extract an INVESTMENT METHODOLOGY from source material about an investor.
You also have background knowledge of well-known investors; you may use it, but every claim must be
something you would defend as this investor's actual approach. Never invent specifics to fill space.

Output ONLY valid JSON (no prose, no markdown fences) matching this schema exactly:
{
  "investor_name": str,
  "summary": str,                          // 2-3 sentences on their core approach
  "mental_models": [
    {"name": str, "description": str, "how_applied": str}
  ],
  "decision_heuristics": [
    {"rule": str, "example": str}
  ],
  "analysis_dimensions": [str],            // what they examine in a company (e.g. moat, management, balance sheet)
  "circle_of_competence": {
    "understands": [str],                  // sectors/situations they actively engage
    "avoids": [str]                        // what they explicitly stay away from
  },
  "red_flags": [str],                      // what makes them pass or sell
  "evidence": {
    "mental_models": "strong" | "weak" | "insufficient",
    "decision_heuristics": "strong" | "weak" | "insufficient",
    "circle_of_competence": "strong" | "weak" | "insufficient",
    "overall": "strong" | "weak" | "insufficient"
  },
  "notes": str                             // caveats; explicitly flag thin or guessed areas
}

Rules:
- If the material (and your knowledge) do not support a field, return an empty list/string and mark
  that section's evidence "insufficient". Do NOT fabricate.
- Prefer concrete, falsifiable rules over vague platitudes.
- Investment methodology ONLY. Ignore personality, speaking style, and biography.
- "overall" should reflect the weakest load-bearing sections, not an average."""


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()


def extract(investor: str, source_brief: str) -> Dict:
    """Turn the source brief into the structured framework the runtime consumes."""
    strong = ChatAnthropic(model_name=STRONG_MODEL, temperature=0, max_tokens=4096)
    user = (
        f"INVESTOR: {investor}\n\n"
        f"SOURCE MATERIAL (may be thin — be honest if so):\n"
        f"{source_brief or '(no material retrieved)'}"
    )
    msg = strong.invoke(
        [SystemMessage(content=EXTRACTION_SYSTEM), HumanMessage(content=user)]
    )
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    return json.loads(_strip_fences(text))


# =============================================================================
# Quality, rendering, persistence
# =============================================================================
def compute_quality(framework: Dict) -> Dict:
    """
    Derive quality from the model's own evidence markers + structural completeness.
    `confidence` is what the lead agent's _effective_perspective_weight() consumes
    to shrink a thin perspective's influence on the final score.
    """
    ev = framework.get("evidence", {}) or {}
    overall = ev.get("overall", "insufficient")
    confidence = {"strong": "High", "weak": "Medium", "insufficient": "Low"}.get(
        overall, "Low"
    )
    mm = len(framework.get("mental_models", []) or [])
    dh = len(framework.get("decision_heuristics", []) or [])
    return {
        "evidence": ev,
        "confidence": confidence,
        "mental_models_count": mm,
        "decision_heuristics_count": dh,
        # gate on extractable methodology, NOT raw source count
        "passes_gate": (mm >= 3 and dh >= 3 and overall != "insufficient"),
    }


def render_skill_md(framework: Dict, quality: Dict) -> str:
    """Human-readable companion for vetting. The runtime reads framework.json, not this."""
    f = framework
    lines: List[str] = []
    lines.append(f"# Investment Perspective: {f.get('investor_name', 'Unknown')}")
    lines.append("")
    lines.append(f"> Confidence: **{quality['confidence']}** · "
                 f"Passes gate: **{quality['passes_gate']}** · "
                 f"Evidence (overall): **{f.get('evidence', {}).get('overall', 'n/a')}**")
    lines.append(">")
    lines.append("> _Distilled model of stated heuristics — not the investor's judgment. "
                 "Review before use._")
    lines.append("")
    if f.get("summary"):
        lines.append("## Summary")
        lines.append(f["summary"])
        lines.append("")

    if f.get("mental_models"):
        lines.append("## Mental Models")
        for m in f["mental_models"]:
            lines.append(f"### {m.get('name', '')}")
            if m.get("description"):
                lines.append(m["description"])
            if m.get("how_applied"):
                lines.append(f"*Applied:* {m['how_applied']}")
            lines.append("")

    if f.get("decision_heuristics"):
        lines.append("## Decision Heuristics")
        for h in f["decision_heuristics"]:
            ex = f" — _e.g._ {h['example']}" if h.get("example") else ""
            lines.append(f"- **{h.get('rule', '')}**{ex}")
        lines.append("")

    if f.get("analysis_dimensions"):
        lines.append("## Analysis Dimensions")
        lines.append(", ".join(f["analysis_dimensions"]))
        lines.append("")

    coc = f.get("circle_of_competence", {}) or {}
    if coc.get("understands") or coc.get("avoids"):
        lines.append("## Circle of Competence")
        if coc.get("understands"):
            lines.append(f"**Engages:** {', '.join(coc['understands'])}")
        if coc.get("avoids"):
            lines.append(f"**Avoids:** {', '.join(coc['avoids'])}")
        lines.append("")

    if f.get("red_flags"):
        lines.append("## Red Flags (pass / sell triggers)")
        for r in f["red_flags"]:
            lines.append(f"- {r}")
        lines.append("")

    if f.get("notes"):
        lines.append("## Notes & Caveats")
        lines.append(f["notes"])
        lines.append("")

    return "\n".join(lines)


def normalize_key(investor: str) -> str:
    """'Serenity@aleabitoreddit' -> 'serenity-aleabitoreddit'; 'Charlie Munger' -> 'charlie-munger'."""
    return re.sub(r"[^a-z0-9]+", "-", investor.lower()).strip("-")


def save(investor: str, key: str, framework: Dict, quality: Dict,
         method: str = "offline-rag-v1") -> str:
    out_dir = os.path.join(CACHE_DIR, key)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "framework.json"), "w", encoding="utf-8") as fh:
        json.dump(framework, fh, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "SKILL.md"), "w", encoding="utf-8") as fh:
        fh.write(render_skill_md(framework, quality))

    metadata = {
        "investor_name": investor,
        "normalized_key": key,
        "distillation_date": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "quality_metrics": quality,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    return out_dir


# =============================================================================
# CLI
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Distill an investor perspective (offline).")
    ap.add_argument("investor", nargs="?",
                    help='Name or handle, e.g. "Charlie Munger". '
                         'Optional when --posts is given (derived from the file).')
    ap.add_argument("--posts", metavar="PATH",
                    help="Distill from a pre-fetched X-posts JSON file instead of web search.")
    ap.add_argument("--key", help="Cache key override (default: normalized name)")
    ap.add_argument("--review", action="store_true",
                    help="Print the generated SKILL.md after distilling, for vetting")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")

    # ---- resolve identity + source --------------------------------------------
    if args.posts:
        if not os.path.exists(args.posts):
            raise SystemExit(f"Posts file not found: {args.posts}")
        with open(args.posts, encoding="utf-8") as fh:
            header = json.load(fh)
        handle = header.get("username", "")
        investor = args.investor or f"{header.get('display_name') or handle} (@{handle})"
        method = "offline-x-posts-v1"
        source_desc = f"X posts file ({header.get('count', '?')} posts) -> {args.posts}"
    else:
        if not args.investor:
            raise SystemExit("Provide an investor name/handle, or --posts <file.json>.")
        investor = args.investor
        method = "offline-rag-v1"
        source_desc = "web search (Tavily)"

    key = args.key or normalize_key(investor)
    print(f"Distilling '{investor}'  ->  {CACHE_DIR}/{key}/")
    print(f"  source: {source_desc}")

    print("  1/3  retrieving source material ...")
    brief = load_posts_brief(args.posts) if args.posts else retrieve(investor)
    if not brief.strip():
        print("  ! no usable material retrieved — perspective will be low quality.")

    print("  2/3  extracting framework ...")
    framework = extract(investor, brief)

    print("  3/3  scoring + saving ...")
    quality = compute_quality(framework)
    path = save(investor, key, framework, quality, method=method)

    print(f"\n  saved -> {path}")
    print(f"  confidence: {quality['confidence']}  |  passes gate: {quality['passes_gate']}")
    print(f"  mental models: {quality['mental_models_count']}  |  "
          f"heuristics: {quality['decision_heuristics_count']}")
    if not quality["passes_gate"]:
        print("  ! below quality gate — review before use, or skip this investor.")

    if args.review:
        print("\n" + "=" * 60 + "\n")
        with open(os.path.join(path, "SKILL.md"), encoding="utf-8") as fh:
            print(fh.read())


if __name__ == "__main__":
    main()