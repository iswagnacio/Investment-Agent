# Deploy the 500D web demo

**Status:** The source snapshot and local checks are prepared. The app has not
been published, and no public URL has been verified. A Streamlit Community Cloud
sign-in, repository access, secrets configuration, and a logged-out smoke test
remain account actions.

For a teammate running the ZIP locally, start with [DEMO_HANDOFF.md](DEMO_HANDOFF.md).

## 1. Make a clean source repository

The existing `iswagnacio/Investment-Agent` repository is public, and its older Git
history may contain personal holdings. Do not deploy or copy that history as a
sanitized source. The ZIP below contains approved web source files, a checksum
manifest, and blank secret placeholders. It contains no `.git`, `.env`,
`holdings.csv`, cache, or research records.

From the **original checkout**, build the ZIP:

```bash
python3.11 scripts/package_demo.py
```

Create a **new, empty private GitHub repository** in your account (for example,
`Investment-Agent-500D-demo`). In a separate directory, unpack the ZIP and push
only the extracted snapshot. Replace `YOUR_GITHUB_USER` and `NEW_REPO_NAME` with
the private repository you created:

```bash
mkdir -p ../Investment-Agent-500D-demo-source
cd ../Investment-Agent-500D-demo-source
unzip ../Investment/dist/Investment-Agent-500D-demo.zip
git init -b 500D
git add -A
git commit -m "Prepare 500D web demo"
git remote add origin git@github.com:YOUR_GITHUB_USER/NEW_REPO_NAME.git
git push -u origin 500D
```

The `unzip` path assumes the new directory is next to the original `Investment`
checkout; adjust that path if your directories differ. Inspect
`DEPLOYMENT_MANIFEST.json` and `git status` before pushing. Keep the new source
repository private. The generated ZIP is ignored by the original checkout.

## 2. Configure Streamlit Community Cloud

1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and grant
   it access to the new private GitHub repository. Repository admin access may be
   needed for the initial connection.
2. Select **Create app** / **Deploy an app**, then choose the new repository,
   branch `500D`, and entrypoint `streamlit_app.py` at the repository root.
3. In **Advanced settings**, choose **Python 3.11**. The verified local runtime
   uses 3.11; the current Community Cloud default may differ. The root
   `requirements.txt` installs `.[web]`, whose compatible versions are specified
   in `pyproject.toml`.
4. In the deployment's **Secrets** field, enter the keys below as root-level TOML
   entries. Fill the required Anthropic value with your own credential in the
   dashboard. Add optional provider keys only when you use them. Do not place
   these keys under a TOML table such as `[api_keys]`, and do not commit a
   populated `.streamlit/secrets.toml`.

   ```toml
   ANTHROPIC_API_KEY = "" # Replace the blank value in the dashboard.
   # Optional extra news providers:
   # FINNHUB_API_KEY = "..."
   # NEWS_API_KEY = "..."
   # ALPHA_VANTAGE_API_KEY = "..."
   ```

5. Deploy and wait for the build and app health checks. If the private-repository
   app is initially private, set **App settings → Sharing → This app is public and
   searchable**. This changes app access; it does not require making the source
   repository public.
6. Open the resulting URL in a private/logged-out browser window. Confirm that
   neither a Streamlit login nor your GitHub account is needed. Record the real
   URL only after this check.

Community Cloud's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[dependency rules](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[secrets rules](https://docs.streamlit.io/develop/concepts/connections/secrets-management),
and [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
describe these settings. Root-level Streamlit secrets are exposed as environment
variables, which the existing agents read with `os.getenv`.

## 3. Verify the live app before presenting

- Open **Stock Research**, enter `AAPL`, and submit. A real request uses paid
  Anthropic calls and live market/news sources. Check generation time, scores,
  source dates, and any pipeline availability notes. Missing or stale sources
  should produce an honest unavailable result, not invented figures.
- Open **Portfolio Assistant**. Ask “What's in my portfolio?” and then “Tell me
  about my largest holding.” Confirm the second answer uses the first turn and
  describes the synthetic holdings by share count. Click **Reset conversation**
  and confirm the context clears. A separate browser session should have its own
  history. The hosted portfolio is read-only and cannot modify the synthetic CSV.
- Try an attempted write request. It should refuse to change holdings. Confirm
  no private local holdings or secrets are accessible through the app.
- Check provider credit and quotas before live Q&A, and warm the app shortly
  before the presentation. A successful offline smoke test does not prove live
  provider availability or model quality.

Conversation history is session-local and is lost when a browser session or
hosted process restarts. The demo does not store browser questions or portfolio
state in a database or versioned research record. Cloud filesystem contents are
ephemeral; do not rely on local cache persistence.

## Verification scope

The exported snapshot is tested independently with synthetic, offline model
responses and the existing ReAct read tool. Python 3.11 Linux wheel resolution
was checked, but a native Linux runtime test was unavailable because the local
Docker daemon was not running. A real hosted build, both live workflows, and a
public logged-out URL remain to be verified after account setup.
