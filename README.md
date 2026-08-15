# Customer Churn Prediction API

A production-ready Flask REST API that predicts customer churn using a trained Random Forest classifier. The project also includes a batch scoring script for processing large customer datasets offline.

---

## Project Structure

```
customer-churn-api/
├── app/
│   ├── __init__.py       # Package initialiser
│   ├── main.py           # Flask API (POST /predict)
│   └── utils.py          # Artefact loading & inference helpers
├── test_data/
│   ├── all_customers.csv # Customer dataset for batch scoring
│   └── sample_input.json # Example single-record JSON payload
├── logs/                 # Auto-created by batch.py
│   └── batch_log.txt
├── batch.py              # Batch scoring script
├── train.py              # Model training script
├── requirements.txt      # Python dependencies
└── README.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

> **Must be run before starting the API.**

```bash
python train.py
```

This generates `app/model.pkl` and `app/transformer.pkl`.

---

## Running the API

```bash
python app/main.py
```

The server starts on **http://127.0.0.1:8000**.

### Health Check

```bash
curl http://127.0.0.1:8000/
```

**Response:**
```json
{"status": "ok", "message": "Churn Prediction API is running."}
```

### Predict Endpoint

**`POST /predict`**

Send a JSON body with the following fields:

| Field | Type | Example |
|---|---|---|
| `tenure` | int | `2` |
| `MonthlyCharges` | float | `70.7` |
| `TotalCharges` | float | `151.65` |
| `SeniorCitizen` | int (0/1) | `0` |
| `gender` | string | `"Female"` |
| `Partner` | string | `"No"` |
| `Dependents` | string | `"No"` |
| `PhoneService` | string | `"Yes"` |
| `MultipleLines` | string | `"No"` |
| `InternetService` | string | `"Fiber optic"` |
| `OnlineSecurity` | string | `"No"` |
| `OnlineBackup` | string | `"No"` |
| `DeviceProtection` | string | `"No"` |
| `TechSupport` | string | `"No"` |
| `StreamingTV` | string | `"No"` |
| `StreamingMovies` | string | `"No"` |
| `Contract` | string | `"Month-to-month"` |
| `PaperlessBilling` | string | `"Yes"` |
| `PaymentMethod` | string | `"Electronic check"` |

**Example request using the sample file:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d @test_data/sample_input.json
```

**Example response:**

```json
{
    "churn_prediction": 1,
    "churn_probability": 0.8350
}
```

---

## Batch Scoring

With the API running in a separate terminal, run:

```bash
python batch.py
```

This will:
1. Read every row from `test_data/all_customers.csv`
2. Send each record to `POST /predict`
3. Write predictions to **`scored_customers.csv`**
4. Log a full summary report to **`logs/batch_log.txt`**

---

## Maintenance Plan

### Retraining

The model should be retrained **monthly** or whenever churn rate in production deviates more than **±5 percentage points** from the training baseline. To retrain: update `test_data/all_customers.csv` with fresh labelled data, run `python train.py`, and redeploy the updated `.pkl` files. Use a version-controlled filename convention (e.g., `model_v2_2026-09.pkl`) and keep the previous artefact for rollback. Automate retraining via a cron job or CI/CD pipeline trigger. Always evaluate on a held-out test split and compare ROC-AUC against the current production model before promoting a new version. Minimum acceptable ROC-AUC is **0.75**.

### Drift Detection

Monitor **data drift** monthly by comparing feature distributions (mean, std, category frequencies) of incoming API payloads against the training distribution. Use Population Stability Index (PSI) — flag any feature with PSI > 0.2 for investigation. Monitor **concept drift** by tracking prediction distributions and, where labels are available (e.g., after 90 days), computing actual vs predicted churn rates. Log all incoming payloads to a database or data warehouse to enable retrospective analysis. Alert the team via email or Slack if drift thresholds are breached.

### Versioning

All code, training scripts, and serialised artefacts are version-controlled in this Git repository. Use **semantic versioning** for model releases (e.g., `v1.0.0`). Tag each release in Git (`git tag v1.0.0`) and store corresponding `.pkl` artefacts with matching version suffixes in a dedicated `models/` directory or an object storage bucket (e.g., AWS S3). Maintain a `CHANGELOG.md` documenting changes per version. Each deployment should reference an explicit model version so that rollback is a single config change. Never overwrite production artefacts in place without first confirming the new model passes evaluation criteria.
