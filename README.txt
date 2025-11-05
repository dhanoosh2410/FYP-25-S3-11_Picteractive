Picteractive – updated run instructions (client/server split)

What changed
- Frontend config now lives in `client/` but serves the repo root.
- React source stays in top-level `src/`.
- FastAPI lives in `server/` with a dev entrypoint `dev_api.py`.

Run (development)
1) Start the API (Python 3.10+ with requirements installed):
   - From repo root (recommended):
     python server/dev_api.py
   - Or from client using npm script:
     cd client && npm run api

2) Start the Vite dev server:
   cd client
   npm install   (first time)
   npm run dev

   Vite serves the repo root so index.html comes from `./index.html` and
   React code from `./src`. Open http://localhost:5173

Environment
- App reads `.env` at repo root.
- Important keys:
  - VITE_API_BASE=http://localhost:8000
  - DATABASE_URL / other server settings as needed

Build
- Production build outputs to `client/dist`:
  cd client && npm run build

Notes
- CORS is preconfigured for http://localhost:5173
- SQLite DB path defaults to `server/data/app.db`


Terminal 1 (API)

From repo root:
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r server\requirements.txt
uvicorn server.main:app --reload --port 8000 
Terminal 2 (Frontend)

cd client
npm install
npm run dev
Verify

Frontend: http://localhost:5173
API health: http://localhost:8000/api/health