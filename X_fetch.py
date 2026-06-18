"""
x_post_fetcher.py
=================
Standalone module that fetches all available posts for a given X (Twitter) user,
up to the X API v2 hard ceiling of 3,200 posts per user timeline.

Intended use: one-time ingestion of an investor's posts as distiller source data.
Because the X API is pay-per-use (reads cost ~$0.005 each as of 2026), results are
cached to disk and reused. You pay once per user, then read from cache forever.

Auth:   Set the X_BEARER_TOKEN environment variable (OAuth 2.0 app-only bearer token).
        Read-only public-timeline access is all this needs.

Usage (library):
    from x_post_fetcher import XPostFetcher

    fetcher = XPostFetcher()
    result = fetcher.fetch_all_posts("WarrenBuffett")          # cache-gated full pull
    for post in result["posts"]:
        print(XPostFetcher.full_text(post))                    # handles long-form posts

    fetcher.update("WarrenBuffett")                            # cheap incremental top-up

Usage (CLI):
    export X_BEARER_TOKEN="..."
    python x_post_fetcher.py WarrenBuffett
    python x_post_fetcher.py WarrenBuffett --exclude retweets,replies --refresh
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("x_post_fetcher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "https://api.x.com/2"
TIMELINE_HARD_CAP = 3200          # X API ceiling for the user-timeline endpoint
MAX_RESULTS_PER_PAGE = 100        # endpoint maximum per request
MAX_RATE_LIMIT_WAIT = 900         # seconds; refuse to sleep longer than this (15 min)

# Fields worth pulling for a distiller. note_tweet carries the FULL text of
# long-form (>280 char) posts, which would otherwise be truncated in `text`.
DEFAULT_TWEET_FIELDS = ",".join([
    "created_at",
    "text",
    "note_tweet",
    "lang",
    "public_metrics",
    "referenced_tweets",   # lets you tell originals from retweets/replies/quotes
    "conversation_id",
    "entities",
])


class XPostFetcherError(Exception):
    """Raised for unrecoverable fetch errors (auth, quota, bad user, etc.)."""


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------
class XPostFetcher:
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        cache_dir: str = "data/x_cache",
        timeout: int = 30,
        max_retries: int = 5,
    ):
        token = bearer_token or os.getenv("X_BEARER_TOKEN")
        if not token:
            raise XPostFetcherError(
                "No bearer token. Set X_BEARER_TOKEN or pass bearer_token=..."
            )
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}"})

    # ----- low-level HTTP with rate-limit handling -------------------------
    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        for attempt in range(1, self.max_retries + 1):
            resp = self.session.get(url, params=params, timeout=self.timeout)

            if resp.status_code == 429:
                wait = self._seconds_until_reset(resp)
                if wait > MAX_RATE_LIMIT_WAIT:
                    raise XPostFetcherError(
                        f"Rate limit / quota reset is {wait}s away "
                        f"(> {MAX_RATE_LIMIT_WAIT}s). You may be out of credits or "
                        f"monthly read quota. Aborting rather than sleeping."
                    )
                logger.warning("429 received; sleeping %ss (attempt %d/%d)",
                               wait, attempt, self.max_retries)
                time.sleep(wait)
                continue

            if resp.status_code == 401:
                raise XPostFetcherError("401 Unauthorized: check your bearer token.")
            if resp.status_code == 403:
                raise XPostFetcherError(
                    "403 Forbidden: your access tier may not permit this endpoint."
                )
            if resp.status_code == 404:
                raise XPostFetcherError("404 Not Found: user does not exist or is suspended.")
            if resp.status_code >= 500:
                backoff = min(2 ** attempt, 60)
                logger.warning("Server error %s; backing off %ss", resp.status_code, backoff)
                time.sleep(backoff)
                continue

            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise XPostFetcherError(f"HTTP error: {exc} -- {resp.text[:300]}")

            data = resp.json()
            # Partial errors can co-exist with data; log but don't abort.
            if "errors" in data and not data.get("data"):
                raise XPostFetcherError(f"API returned errors: {data['errors']}")
            if "errors" in data:
                logger.warning("Partial errors in response: %s", data["errors"])
            return data

        raise XPostFetcherError(f"Exhausted {self.max_retries} retries for {url}")

    @staticmethod
    def _seconds_until_reset(resp: requests.Response) -> int:
        # Prefer the precise reset header; fall back to Retry-After, then 60s.
        reset = resp.headers.get("x-rate-limit-reset")
        if reset:
            delta = int(reset) - int(time.time())
            return max(delta + 2, 1)
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            return int(retry_after) + 1
        return 60

    # ----- public API ------------------------------------------------------
    def resolve_user_id(self, username: str) -> Dict[str, str]:
        """Resolve a @handle to its numeric user id (the timeline endpoint needs the id)."""
        username = username.lstrip("@")
        url = f"{API_BASE}/users/by/username/{username}"
        data = self._request(url, {"user.fields": "id,name,username"})
        if "data" not in data:
            raise XPostFetcherError(f"Could not resolve user '{username}': {data}")
        logger.info("Resolved @%s -> id %s", username, data["data"]["id"])
        return data["data"]

    def fetch_all_posts(
        self,
        username: str,
        exclude: Optional[str] = None,      # e.g. "retweets" or "retweets,replies"
        start_time: Optional[str] = None,   # ISO 8601, e.g. "2023-01-01T00:00:00Z"
        end_time: Optional[str] = None,
        max_posts: int = TIMELINE_HARD_CAP,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Fetch up to `max_posts` (capped at 3,200) for `username`.
        Returns the full cache record. Uses the on-disk cache unless force_refresh.

        For distillation you usually want exclude="retweets" so the corpus is the
        user's own words, not content they amplified. Default is None = literally all.
        """
        username = username.lstrip("@")
        cached = None if force_refresh else self._load_cache(username)
        if cached is not None:
            logger.info("Cache hit for @%s (%d posts, fetched %s)",
                        username, cached["count"], cached["fetched_at"])
            return cached

        user = self.resolve_user_id(username)
        posts = self._fetch_timeline(
            user["id"], exclude=exclude, start_time=start_time,
            end_time=end_time, max_posts=max_posts,
        )
        record = self._build_record(username, user, posts)
        self._save_cache(username, record)
        logger.info("Fetched and cached %d posts for @%s", record["count"], username)
        return record

    def update(self, username: str, exclude: Optional[str] = None) -> Dict[str, Any]:
        """
        Cheap incremental top-up: fetch only posts newer than the newest cached id,
        prepend them, and re-cap at 3,200. No-op-ish cost if nothing is new.
        """
        username = username.lstrip("@")
        cached = self._load_cache(username)
        if cached is None:
            logger.info("No cache for @%s; doing a full fetch instead.", username)
            return self.fetch_all_posts(username, exclude=exclude)

        new_posts = self._fetch_timeline(
            cached["user_id"], since_id=cached.get("newest_id"), exclude=exclude,
        )
        if not new_posts:
            logger.info("@%s is up to date; no new posts.", username)
            return cached

        merged = (new_posts + cached["posts"])[:TIMELINE_HARD_CAP]
        record = self._build_record(
            username, {"id": cached["user_id"], "name": cached.get("display_name", ""),
                       "username": username}, merged,
        )
        self._save_cache(username, record)
        logger.info("Added %d new posts for @%s (total %d).",
                    len(new_posts), username, record["count"])
        return record

    # ----- internals -------------------------------------------------------
    def _fetch_timeline(
        self,
        user_id: str,
        since_id: Optional[str] = None,
        exclude: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        max_posts: int = TIMELINE_HARD_CAP,
    ) -> List[Dict[str, Any]]:
        url = f"{API_BASE}/users/{user_id}/tweets"
        params: Dict[str, Any] = {
            "max_results": MAX_RESULTS_PER_PAGE,
            "tweet.fields": DEFAULT_TWEET_FIELDS,
        }
        if since_id:
            params["since_id"] = since_id
        if exclude:
            params["exclude"] = exclude
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        posts: List[Dict[str, Any]] = []
        next_token: Optional[str] = None
        cap = min(max_posts, TIMELINE_HARD_CAP)

        while len(posts) < cap:
            if next_token:
                params["pagination_token"] = next_token
            data = self._request(url, params)
            batch = data.get("data", [])
            if not batch:
                break
            posts.extend(batch)
            logger.info("  fetched %d posts so far...", len(posts))
            next_token = data.get("meta", {}).get("next_token")
            if not next_token:
                break

        return posts[:cap]

    @staticmethod
    def _build_record(username: str, user: Dict[str, Any],
                      posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "username": username,
            "user_id": user["id"],
            "display_name": user.get("name", ""),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(posts),
            "newest_id": posts[0]["id"] if posts else None,  # timeline is newest-first
            "posts": posts,
        }

    # ----- cache helpers ---------------------------------------------------
    def _cache_path(self, username: str) -> str:
        safe = username.lstrip("@").lower()
        return os.path.join(self.cache_dir, f"{safe}.json")

    def _load_cache(self, username: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(username)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read cache %s (%s); will refetch.", path, exc)
            return None

    def _save_cache(self, username: str, record: Dict[str, Any]) -> None:
        os.makedirs(self.cache_dir, exist_ok=True)
        path = self._cache_path(username)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)

    # ----- text helper -----------------------------------------------------
    @staticmethod
    def full_text(post: Dict[str, Any]) -> str:
        """Return the complete post text, preferring note_tweet for long-form posts."""
        note = post.get("note_tweet")
        if note and note.get("text"):
            return note["text"]
        return post.get("text", "")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def fetch_user_posts(username: str, **kwargs) -> List[Dict[str, Any]]:
    """One-liner that returns just the list of post objects for `username`."""
    return XPostFetcher().fetch_all_posts(username, **kwargs)["posts"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch all posts (up to the 3,200 cap) for a given X user."
    )
    parser.add_argument("username", help="X handle, with or without the leading @")
    parser.add_argument("--exclude", help="Comma list: retweets, replies", default=None)
    parser.add_argument("--max", type=int, default=TIMELINE_HARD_CAP,
                        help="Max posts to fetch (capped at 3,200)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cache and refetch from the API")
    parser.add_argument("--update", action="store_true",
                        help="Incremental top-up of an existing cache")
    parser.add_argument("--cache-dir", default="data/x_cache")
    args = parser.parse_args()

    try:
        fetcher = XPostFetcher(cache_dir=args.cache_dir)
        if args.update:
            record = fetcher.update(args.username, exclude=args.exclude)
        else:
            record = fetcher.fetch_all_posts(
                args.username, exclude=args.exclude,
                max_posts=args.max, force_refresh=args.refresh,
            )
    except XPostFetcherError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    print(f"\n@{record['username']} ({record['display_name']}) "
          f"-- {record['count']} posts cached")
    for post in record["posts"][:5]:
        text = XPostFetcher.full_text(post).replace("\n", " ")
        print(f"  [{post.get('created_at', '?')}] {text[:90]}...")


if __name__ == "__main__":
    _main()