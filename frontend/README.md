# Frontend (Vite + Vue 3 + TypeScript)

Install dependencies and run dev server:

```bash
cd frontend
npm install
npm run dev
```

Dev server default: http://localhost:5173

## Configuration

- Backend API base: optionally set in browser via a global before app loads
	- Example (in `index.html` before the script):
		<script>window.VITE_API_BASE = 'http://127.0.0.1:8022'</script>
	- Fallback if not set: `http://127.0.0.1:8022`

## What you’ll see

The dashboard polls the backend `/nowcast` endpoint every 10 seconds and renders:

- Per-symbol card with latest timestamp, price, and heuristic bottom score
- Base model probabilities (xgb, lstm, tf) when available
- Stacking ensemble summary per symbol:
	- prob (stacked probability)
	- threshold (decision threshold)
	- decision ON/OFF badge
	- confidence (distance to threshold)
- A top meta bar showing the currently used stacking method, threshold, and models (from `_stacking_meta` in the `/nowcast` payload)

Notes

- The frontend ignores special keys in the payload (e.g., `_stacking_meta`) when listing symbol cards.
- Poll interval can be adjusted at runtime from the UI; defaults to 10 seconds.
