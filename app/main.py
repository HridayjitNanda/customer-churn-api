"""
app/main.py
-----------
Flask API exposing a /predict endpoint for customer churn prediction.

Usage (from project root):
    python -m app.main
"""

from flask import Flask, request, jsonify
from app.utils import load_artifacts, predict_single

app = Flask(__name__)

# Load artifacts once at startup
transformer, model = load_artifacts()


@app.route("/", methods=["GET"])
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "message": "Churn Prediction API is running."}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict customer churn from a nested JSON payload.

    Expected JSON body:
        {
            "customer": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "tenure": 12,
                ...
            }
        }

    Returns:
        {
            "churn_probability": 0.83,
            "churn_prediction": "Yes"
        }
    """
    payload = request.get_json(silent=True)

    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    if "customer" not in payload:
        return jsonify({
            "error": "Missing 'customer' key. Payload must be {'customer': {...}}"
        }), 422

    customer = payload["customer"]

    try:
        result = predict_single(customer, transformer, model)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
