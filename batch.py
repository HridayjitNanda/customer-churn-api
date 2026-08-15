"""
batch.py
--------
Batch scoring script for the Customer Churn Prediction API.

Reads a customer CSV, sends each row as a nested JSON payload to the
/predict endpoint, collects predictions, writes scored_customers.csv,
and logs summary statistics to logs/batch_log.txt.

Usage:
    # Ensure the Flask API is running first:
    #   python -m app.main
    python batch.py --input test_data/all_customers.csv
"""

import os
import argparse
import logging
import datetime
import requests
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL    = "http://localhost:8000/predict"
OUTPUT_CSV = "scored_customers.csv"
LOG_DIR    = "logs"
LOG_FILE   = os.path.join(LOG_DIR, "batch_log.txt")

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def row_to_customer(row: pd.Series) -> dict:
    """Convert a DataFrame row to the API's nested customer payload."""
    customer = {}
    for col, val in row.items():
        if col == "customerID":
            continue
        # Preserve correct types
        if col in ("SeniorCitizen", "tenure"):
            customer[col] = int(val)
        elif col in ("MonthlyCharges", "TotalCharges"):
            try:
                customer[col] = float(val)
            except (ValueError, TypeError):
                customer[col] = 0.0
        else:
            customer[col] = str(val)
    return customer


def run_batch(input_csv: str):
    start_time = datetime.datetime.now()
    logger.info("=" * 60)
    logger.info("Batch scoring started at %s", start_time.isoformat())
    logger.info("Input  : %s", input_csv)
    logger.info("Output : %s", OUTPUT_CSV)
    logger.info("API URL: %s", API_URL)
    logger.info("=" * 60)

    df = pd.read_csv(input_csv)
    # Coerce TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    total     = len(df)
    successes = 0
    failures  = 0
    results   = []

    for idx, row in df.iterrows():
        customer_id = row.get("customerID", f"row_{idx}")
        payload = {"customer": row_to_customer(row)}

        try:
            response = requests.post(API_URL, json=payload, timeout=10)

            if response.status_code != 200:
                logger.warning("Customer %s | HTTP %d: %s",
                               customer_id, response.status_code, response.text)
                failures += 1
                continue

            data = response.json()
            pred = data["churn_prediction"]
            prob = data["churn_probability"]
            results.append({
                "customerID":        customer_id,
                "churn_prediction":  pred,
                "churn_probability": prob
            })
            successes += 1
            logger.info("Customer %-15s | prediction=%-3s | probability=%.4f",
                        customer_id, pred, prob)

        except requests.exceptions.ConnectionError:
            logger.error("Customer %s | Connection refused -- is the API running?", customer_id)
            failures += 1
        except requests.exceptions.Timeout:
            logger.error("Customer %s | Request timed out.", customer_id)
            failures += 1
        except Exception as exc:
            logger.error("Customer %s | Unexpected error: %s", customer_id, exc)
            failures += 1

    # -- Write output CSV -----------------------------------------------------
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv(OUTPUT_CSV, index=False)
        logger.info("\nScored results written to %s", OUTPUT_CSV)

    # -- Summary statistics ---------------------------------------------------
    end_time  = datetime.datetime.now()
    duration  = (end_time - start_time).total_seconds()
    probs     = [r["churn_probability"] for r in results]
    avg_prob  = sum(probs) / len(probs) if probs else 0.0
    churn_yes = sum(1 for r in results if r["churn_prediction"] == "Yes")

    logger.info("\n" + "=" * 60)
    logger.info("BATCH SUMMARY")
    logger.info("=" * 60)
    logger.info("Completed at       : %s", end_time.isoformat())
    logger.info("Duration           : %.2f seconds", duration)
    logger.info("Total requests     : %d", total)
    logger.info("Successful calls   : %d", successes)
    logger.info("Failed predictions : %d", failures)
    logger.info("Churn = Yes        : %d", churn_yes)
    logger.info("Churn = No         : %d", successes - churn_yes)
    logger.info("Avg churn prob     : %.4f", avg_prob)
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch churn scoring script")
    parser.add_argument(
        "--input",
        default="test_data/all_customers.csv",
        help="Path to input CSV file (default: test_data/all_customers.csv)"
    )
    args = parser.parse_args()
    run_batch(args.input)
