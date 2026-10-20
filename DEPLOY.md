# Deploying the demo to Streamlit Community Cloud

The app opens in **Sample demo** mode, which uses the committed offline review
packages and needs no API key. That is the mode to share with reviewers.

## 1. Push to GitHub

From the `supplier-contract-review` folder (PowerShell):

```powershell
git init
git add .
git status            # check: no .env, no .venv, no runs/ in the list
git commit -m "Supplier contract review demo"
git branch -M main
git remote add origin https://github.com/<you>/supplier-contract-review.git
git push -u origin main
```

`.gitignore` already excludes `.env`, `contract_review/.env`, `.venv/` and `runs/`.
Double-check `git status` before committing: the Gemini key must never be pushed.

## Record live runs before you deploy

The hosted app should open in **Replay** with real recorded runs, so the panel never
depends on your API quota. Locally, in **Live AI** mode:

1. Contract & findings → run each bundled contract (contract_a … contract_e), ~3 Gemini requests each.
2. Procurement request → AI intake assistant → Draft request with AI (1 request).
3. Commercial → Negotiate with CloudSoft → Run full negotiation with AI (~9 requests).

On the free tier (~20 requests/day per model) spread this over two days, or let the model chain
rotate to another model. Each run is saved under `data/recordings/` — commit that folder:

```powershell
git add data/recordings
git commit -m "Recorded live AI runs"
git push
```

## 2. Create the app

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **Create app** → pick the repo, branch `main`, main file `ui/streamlit_app.py`.
3. **Advanced settings** → Python **3.11** (matches local development).
4. Optional, for Live mode only: paste the contents of `.streamlit/secrets.toml.example`
   (with your key) into **Secrets**. Streamlit exposes these as environment variables.
5. Deploy. You get a link like `https://<name>.streamlit.app`.

## 3. Before you send the link

- Free apps sleep after a period without visitors. Open the link yourself shortly
  before the panel does, so it is already awake.
- Click through every page once: Request → Contract & findings → Redlines →
  Approvals → Obligations → Scorecard.
- Send the panel: the app link, the GitHub repo, and a 2–3 minute screen recording
  as a fallback.

## Notes

- Each visitor gets their own session. The procurement request and decision logs are
  written under `runs/`, which is temporary on Streamlit Cloud and resets on restart —
  that is fine for a demo.
- Theme, fonts and toolbar settings live in `.streamlit/config.toml`.
- Local run: `streamlit run ui/streamlit_app.py` from this folder.
