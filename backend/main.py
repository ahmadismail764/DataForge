import os
import sys

# Ensure the backend directory is in the path to fix absolute imports when running from the root directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import uuid
import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import run_classification, run_regression, run_clustering

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="AutoML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory session store: session_id -> DataFrame
sessions: dict[str, pd.DataFrame] = {}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    session_id: str
    task: str  # "classification" | "regression" | "clustering"
    target_column: str | None = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept a .csv or .xlsx file, store it in-memory, return preview + columns."""
    filename = file.filename or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext not in (".csv", ".xlsx"):
        raise HTTPException(status_code=400, detail="Only .csv and .xlsx files are supported.")

    try:
        if ext == ".csv":
            df = pd.read_csv(file.file)
        else:
            df = pd.read_excel(file.file, engine="openpyxl")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    session_id = str(uuid.uuid4())
    sessions[session_id] = df

    preview = df.head(10).fillna("").to_dict(orient="records")
    columns = df.columns.tolist()

    return {
        "session_id": session_id,
        "columns": columns,
        "preview": preview,
        "shape": list(df.shape),
    }


@app.post("/api/train")
async def train_model(req: TrainRequest):
    """Run the automated pipeline and return evaluation metrics."""
    df = sessions.get(req.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a file first.")

    task = req.task.lower()

    try:
        if task == "classification":
            if not req.target_column or req.target_column not in df.columns:
                raise HTTPException(status_code=400, detail="A valid target column is required for classification.")
            result = run_classification(df, req.target_column)

        elif task == "regression":
            if not req.target_column or req.target_column not in df.columns:
                raise HTTPException(status_code=400, detail="A valid target column is required for regression.")
            result = run_regression(df, req.target_column)

        elif task == "clustering":
            result = run_clustering(df)

        else:
            raise HTTPException(status_code=400, detail="Task must be classification, regression, or clustering.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    # Save the best pipeline
    model_id = str(uuid.uuid4())
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    joblib.dump(result["pipeline"], model_path)

    # Remove non-serialisable pipeline from response
    result.pop("pipeline", None)
    result["model_id"] = model_id

    return result


@app.get("/api/download/{model_id}")
async def download_model(model_id: str):
    """Download the serialized model file."""
    model_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found.")

    return FileResponse(
        path=model_path,
        filename=f"model_{model_id}.joblib",
        media_type="application/octet-stream",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", port=5001, reload=True)
