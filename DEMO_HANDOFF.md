# 500D presentation demo: teammate handoff

The exported ZIP is a clean source snapshot of the Streamlit demo. It is **not**
a running app or a published URL. It contains the app, dependency files, blank
credential templates, and [DEPLOYMENT.md](DEPLOYMENT.md) for the Community Cloud
route. The ZIP contains no real API key, personal holdings, Git history, or
research records. The `500D` source branch has the same current app files, but
this public repository's older Git history may still contain holdings. The demo
portfolio uses synthetic data and cannot place trades.

## Run locally (macOS or Linux)

1. Unzip the archive into its own directory and open a terminal there. Use
   **Python 3.11**. From that directory:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python -m pip check
   cp .env.example .env
   ```

2. Open the new `.env` in an editor and set `ANTHROPIC_API_KEY` to a valid key
   supplied privately by the account owner. Optional Finnhub, News API, and
   Alpha Vantage keys add news sources; leave them blank if unavailable. Keep
   `.env` on the presenter's machine and out of chat, email, screen sharing, and
   Git. The app makes paid Anthropic calls when a request is submitted; confirm
   account credit and source availability before presenting.
3. Start the app:

   ```bash
   python -m streamlit run streamlit_app.py --server.address 127.0.0.1
   ```

   Open the local URL printed by Streamlit, usually `http://127.0.0.1:8501`.
   If you edit `.env`, restart the Streamlit process before retrying.

## Presentation walkthrough

1. In **Portfolio Assistant**, ask “What's in my portfolio?” The example has
   synthetic AAPL and MSFT holdings. Ask “Tell me about my largest holding” as
   a follow-up; “largest” means share count in this demo. Use **Reset
   conversation** to demonstrate that session memory clears. This workflow is
   read-only and uses no personal `holdings.csv`.
2. In **Stock Research**, enter `AAPL` and click **Run research**. Review the
   generation time, source dates, scores, analysis, and any data availability
   notes. Missing or stale live data may prevent a report; do not present an
   unavailable result as a current market conclusion.
3. Keep both workflows open before Q&A. Each browser session has its own short
   portfolio memory; refreshing or restarting loses it. Live provider/model
   responses vary, so test the exact walkthrough shortly before the talk.

For a shared public URL, use [DEPLOYMENT.md](DEPLOYMENT.md). The Cloud app is not
published yet; someone with Streamlit and GitHub access must complete those
steps and verify the URL in a logged-out browser. Do not upload this snapshot to
the existing public repository without reviewing its older holdings history.
