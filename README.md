# Iris Species Classifier API

A FastAPI service that serves a scikit-learn model trained on the classic Iris dataset.

**Live API:** _URL added after deployment_ · Interactive docs at `/docs`

## What the model predicts

Given four measurements of an Iris flower (in cm), the model predicts its species:
`setosa`, `versicolor`, or `virginica`.

- Model: `StandardScaler` + `LogisticRegression` pipeline (scikit-learn)
- Data: 150 samples from `sklearn.datasets.load_iris`, 80/20 stratified train/test split
- Test accuracy: **93.3%**
- Exported to `model.pkl` with `joblib`

## Endpoints

| Method | Path       | Description                                    |
|--------|------------|------------------------------------------------|
| GET    | `/health`  | Service status and whether the model loaded    |
| POST   | `/predict` | Predicts the species from the four measurements |
| GET    | `/docs`    | Swagger UI for trying the API in the browser   |

### Example request: `POST /predict`

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

### Example response

```json
{
  "species": "setosa",
  "confidence": 0.9808,
  "probabilities": {
    "setosa": 0.9808,
    "versicolor": 0.0192,
    "virginica": 0.0
  }
}
```

### Example `GET /health` response

```json
{
  "status": "ok",
  "model_loaded": true,
  "error": null
}
```

## Run locally

```bash
git clone https://github.com/24f2005056-hub/iris-fastapi.git
cd iris-fastapi
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Optional: retrain and regenerate model.pkl
python train.py

uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs, or call the API from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Deploy (Render)

The repo includes `render.yaml`. On Render, choose **New → Blueprint**, select this repo, and deploy.
Or create a **Web Service** manually with:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## Project structure

```
├── main.py            # FastAPI app (/health, /predict)
├── train.py           # Trains the model and writes model.pkl
├── model.pkl          # Trained model
├── requirements.txt   # Pinned dependencies
├── render.yaml        # Render deployment config
└── .python-version    # Python 3.11.9 (matches the version that trained model.pkl)
```
