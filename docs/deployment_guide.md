# Deployment Guide

The app self-bootstraps its sample data and trains models on startup, so **no
secrets or external services are required** to deploy. Pick one platform below.

---

## 0. Prerequisites (one-time)
1. Ensure the project runs locally:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
2. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Football Intelligence Scouting Platform"
   git branch -M main
   git remote add origin https://github.com/<your-username>/football-intelligence-scouting-platform.git
   git push -u origin main
   ```
   > `.gitignore` already excludes generated data, models, and `.env`.

---

## 1. Streamlit Community Cloud  ✅ recommended (free)
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → select your repository, branch `main`, main file `app.py`.
3. **Advanced settings → Python version:** 3.11 (or 3.10/3.12).
4. Click **Deploy**. First build installs `requirements.txt` and launches.
5. (Optional) Add secrets under **Settings → Secrets** if you later wire in an API
   (use the keys from `.env.example`). Not needed for the default app.

Your app gets a public `*.streamlit.app` URL. Push to `main` to auto-redeploy.

---

## 2. Hugging Face Spaces (free)
1. Create a new **Space** → SDK: **Streamlit**.
2. Push this repo to the Space's git remote (or upload files).
3. HF detects `requirements.txt` and `app.py` automatically and builds.
4. App serves at `https://huggingface.co/spaces/<user>/<space>`.

> Optional `README.md` front-matter for HF (not required):
> ```yaml
> ---
> title: Football Intelligence Scouting Platform
> emoji: ⚽
> sdk: streamlit
> app_file: app.py
> pinned: false
> ---
> ```

---

## 3. Render (free web service)
1. **New → Web Service**, connect the GitHub repo.
2. **Environment:** Python 3.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:**
   ```bash
   streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
5. Deploy. Render assigns a public URL.

---

## 4. Docker (any host) — optional
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```
```bash
docker build -t scouting-platform .
docker run -p 8501:8501 scouting-platform
```

---

## Using real data in production
1. Add your CSV to `data/raw/` (commit it, or mount/upload on the host).
2. Confirm column names map via `COLUMN_ALIASES` in `src/preprocessing.py`.
3. Redeploy / restart. The loader auto-detects `data/raw/*.csv` ahead of sample data.

## Troubleshooting
| Symptom | Fix |
|--------|-----|
| Build fails on dependencies | Pin a supported Python (3.10–3.12); re-run install. |
| App slow on first load | Expected — models train once (~10–15s) then cache. |
| "No players match filters" | Loosen sidebar filters / raise budget. |
| Read-only filesystem warning | Harmless — artifact caching is best-effort and skipped. |
| Want to reset sample data | Run `python -m src.data_loader` locally and redeploy. |
