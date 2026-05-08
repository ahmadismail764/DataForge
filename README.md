# Data Forge

Data Forge is an end-to-end AutoML web application for non-technical users. It provides a simple frontend to upload tabular data, choose an ML task, train multiple models automatically, compare results, and download the best pipeline.

## Features

- Upload datasets in `.csv` or `.xlsx` format
- Preview uploaded data and columns
- Choose task type: classification, regression, or clustering
- Automatic preprocessing (missing values, encoding, scaling, and class balancing where needed)
- Train multiple algorithms per task and select the best performer
- **NEW:** Interactive Plotly.js visualizations for evaluating models (Confusion Matrix, Feature Importance, PCA Clusters, etc.)
- Download trained model pipeline as `.joblib`

## Tech Stack

- **Backend:** FastAPI
- **Frontend:** Vanilla HTML/CSS/JavaScript + Plotly.js
- **ML:** pandas, scikit-learn, imbalanced-learn
- **Model serialization:** joblib

## Project Structure

```text
.
├── backend/
│   ├── main.py
│   ├── pipeline.py
│   └── models/
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
├── requirements.txt
└── README.md
```

## Quick Start

### 1) Clone and enter the project

```bash
git clone <your-repo-url>
cd Project
```

### 2) Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run backend API

```bash
# Using uvicorn:
uvicorn backend.main:app --reload --port 5001

# Or run the script directly:
python backend/main.py
```

### 5) Open frontend

Open `frontend/index.html` in your browser, or serve it locally:

```bash
python3 -m http.server 5500
```

Then navigate to `http://localhost:5500`.

### 6) Launch the notebook (optional)

```bash
jupyter notebook notebook.ipynb
```

The notebook provides the same AutoML pipeline with richer inline visualizations — ideal for exploration and review.

## API Endpoints

- `POST /api/upload` - Upload dataset and get preview
- `POST /api/train` - Train and evaluate based on selected task
- `GET /api/download/{model_id}` - Download trained model pipeline

Interactive API docs are available at:

- `http://localhost:5001/docs`

## Development Notes

- CORS is configured to allow all origins for local development.
- Trained models are saved under `backend/models/` and are gitignored.

## Contributing

Please read `CONTRIBUTING.md` before opening pull requests.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
