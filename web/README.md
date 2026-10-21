# NVMe Health Core
Drive-failure predictor + fleet dashboard. Model trains in Python, ships as JSON, runs in the browser — no backend.

## Retrain (optional; artifacts are already in web/public/model)
    cd ml && pip install pandas scikit-learn && python train.py

## Run
    cd web && npm install && npm run dev

## Deploy (Vercel)
Import the repo, set **Root Directory = web**, deploy. No env vars needed.
