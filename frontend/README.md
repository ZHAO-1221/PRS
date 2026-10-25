# Mobile frontend

React + Vite mobile web app for the dog behaviour recognition system. It talks to the FastAPI service in
`pet_recognition_system/app/server.py`; a mock mode runs the same screens on in-memory sample data.

```bash
cd frontend
npm install
npm run dev        # real API: start the backend first (see below); /api is proxied to localhost:8000
npm run dev:mock   # sample data, no backend needed
npm test           # summary logic tests
npm run build      # static build in dist/
```

Backend, in another terminal (see `pet_recognition_system/README.md`, "Mobile app backend"):

```bash
cd pet_recognition_system
PET_DATA_ROOT=<data-root> uvicorn app.server:app --port 8000
```

The dev server also prints a LAN address: a phone on the same Wi-Fi can open it, and its `/api` calls go
through the same proxy.

| Path | What it is |
|---|---|
| `src/api.js` | All data access (`listClips`, `getClip`, `uploadClip`, `getJob`, `getModelMetrics`) over `fetch('/api/...')`; switches to `src/mock/api.js` when `VITE_MOCK=1`. |
| `src/mock/` | Mock API and sample clips; `makeResult()` builds `predict_video`-shaped results. |
| `src/lib/summary.js` | Turns results into what the screens show (dominant label, day totals, week rows, probabilities). |
| `src/pages/` | Story, Upload, Analyze, Result, Week. |
| `src/components/ClipDeck.jsx` | The stacked, swipeable cover cards. |
| `prototype/`, `prototype.html` | The approved design prototype these screens were built from. |
