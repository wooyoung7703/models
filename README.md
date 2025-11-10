# Models (monorepo)

This repository contains a minimal scaffold for a Python backend and a Vue 3 + TypeScript frontend.

Structure:

- backend/: FastAPI app
- frontend/: Vite + Vue 3 + TypeScript app

Quick start (backend):

```bash
# from repo root
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

Quick start (frontend):

```bash
cd frontend
npm install
npm run dev
```

Notes:
- These are minimal starter files to get you going. Restrict CORS and pin production dependency versions before deploying.
# models