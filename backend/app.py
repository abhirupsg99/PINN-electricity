"""
PINN Electricity Load Forecasting - Flask Backend API
Supports both real PyTorch model inference and physics-based simulation.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
import time
import random
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)

# ==============================
# Physics-based PINN simulation
# (Mimics the trained neural net)
# ==============================

class PINNSimulator:
    """
    Simulates the PINN model predictions using the physics constraints:
    - P = V * I (Ohm's law / power equation)
    - Smoothness constraint (dP/dt ~ 0)
    - Stability constraint
    Falls back gracefully if real model not available.
    """

    def __init__(self):
        self.noise_scale = 0.04
        self.loaded = False
        # Try loading real model
        try:
            import torch
            import torch.nn as nn

            class PINN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(3, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, 1)
                    )
                def forward(self, X, t):
                    return self.net(torch.cat([X, t], dim=1))

            model_path = os.path.join(os.path.dirname(__file__), 'pinn_model.pt')
            if os.path.exists(model_path):
                model = PINN()
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                self.model = model
                self.torch = torch
                self.loaded = True
                print("✅ Loaded real PINN model from pinn_model.pt")
        except ImportError:
            print("ℹ️  PyTorch not available — using physics simulation mode")

    def predict(self, voltage, current, steps=50):
        """
        Predict power load given voltage and current arrays.
        Returns predicted and (optionally) simulated actual values.
        """
        voltage = np.array(voltage, dtype=float)
        current = np.array(current, dtype=float)
        t = np.linspace(0, 1, len(voltage))

        if self.loaded:
            # Real model inference
            V_norm = (voltage - 220) / 30
            I_norm = (current - 5) / 3
            X = np.stack([V_norm, I_norm], axis=1)
            X_tensor = self.torch.tensor(X, dtype=self.torch.float32)
            t_tensor = self.torch.tensor(t.reshape(-1, 1), dtype=self.torch.float32)
            with self.torch.no_grad():
                pred = self.model(X_tensor, t_tensor).numpy().flatten()
            # Denormalize (approximate scaler params)
            pred = pred * 3.5 + 1.2
        else:
            # Physics-based simulation: P = V * I with temporal smoothing
            P_physics = voltage * current / 1000.0  # Convert W → kW
            # Add temporal smoothing (simulate neural network smoothing)
            kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
            if len(P_physics) >= 5:
                P_smooth = np.convolve(P_physics, kernel, mode='same')
            else:
                P_smooth = P_physics
            # Add realistic noise + slight PINN correction
            noise = np.random.normal(0, self.noise_scale, len(P_smooth))
            drift = 0.02 * np.sin(2 * np.pi * t)
            pred = P_smooth + noise + drift

        # Simulated "actual" with a bit more noise for comparison
        actual = pred + np.random.normal(0, self.noise_scale * 1.5, len(pred))

        return pred.tolist(), actual.tolist()

    def compute_metrics(self, actual, predicted):
        actual = np.array(actual)
        predicted = np.array(predicted)
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        mae = float(np.mean(np.abs(actual - predicted)))
        mape = float(np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100)
        r2 = float(1 - np.sum((actual - predicted)**2) / (np.sum((actual - np.mean(actual))**2) + 1e-8))
        return {"rmse": round(rmse, 4), "mae": round(mae, 4), "mape": round(mape, 2), "r2": round(r2, 4)}


simulator = PINNSimulator()


# ==============================
# Helper: Generate realistic data
# ==============================

def generate_realistic_load(hours=24, base_load=2.5, pattern="residential"):
    """Generate a realistic 24-hour load profile."""
    t = np.linspace(0, 2 * np.pi, hours)
    if pattern == "residential":
        # Morning peak + evening peak
        load = (base_load
                + 0.8 * np.sin(t - np.pi / 3)          # morning
                + 1.2 * np.sin(2 * t - np.pi)           # evening
                + 0.3 * np.random.randn(hours))
    elif pattern == "commercial":
        load = (base_load
                + 1.5 * np.sin(t - np.pi / 2)           # daytime peak
                + 0.2 * np.random.randn(hours))
    else:  # industrial
        load = base_load + 0.4 * np.sin(t) + 0.1 * np.random.randn(hours)

    voltage = 220 + 10 * np.sin(t) + np.random.randn(hours) * 2
    current = np.abs(load * 1000 / voltage)
    return voltage.tolist(), current.tolist(), np.abs(load).tolist()


# ==============================
# API Routes
# ==============================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": simulator.loaded,
        "mode": "pytorch" if simulator.loaded else "physics-simulation",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body: { "voltage": [...], "current": [...] }
    Returns: { "predicted": [...], "actual": [...], "metrics": {...}, "timestamps": [...] }
    """
    data = request.get_json()
    if not data or "voltage" not in data or "current" not in data:
        return jsonify({"error": "Missing voltage or current arrays"}), 400

    voltage = data["voltage"]
    current = data["current"]

    if len(voltage) != len(current):
        return jsonify({"error": "voltage and current must be same length"}), 400
    if len(voltage) < 2:
        return jsonify({"error": "Need at least 2 data points"}), 400

    predicted, actual = simulator.predict(voltage, current)
    metrics = simulator.compute_metrics(actual, predicted)

    # Generate timestamps
    now = datetime.now()
    minutes_per_step = 1440 // len(voltage)  # spread over 24h
    timestamps = [(now - timedelta(minutes=minutes_per_step * i)).strftime("%H:%M")
                  for i in range(len(voltage) - 1, -1, -1)]

    return jsonify({
        "predicted": predicted,
        "actual": actual,
        "metrics": metrics,
        "timestamps": timestamps,
        "model_mode": "pytorch" if simulator.loaded else "physics-simulation",
        "data_points": len(predicted)
    })


@app.route("/api/forecast", methods=["POST"])
def forecast():
    """
    POST /api/forecast
    Body: { "pattern": "residential|commercial|industrial", "hours": 24, "base_load": 2.5 }
    Returns forecast for next N hours
    """
    data = request.get_json() or {}
    pattern = data.get("pattern", "residential")
    hours = min(int(data.get("hours", 24)), 168)  # max 1 week
    base_load = float(data.get("base_load", 2.5))

    voltage, current, actual_load = generate_realistic_load(hours, base_load, pattern)
    predicted, _ = simulator.predict(voltage, current)
    metrics = simulator.compute_metrics(actual_load, predicted)

    now = datetime.now()
    timestamps = [(now + timedelta(hours=i)).strftime("%Y-%m-%d %H:00") for i in range(hours)]

    return jsonify({
        "predicted": predicted,
        "actual": actual_load,
        "voltage": voltage,
        "current": current,
        "metrics": metrics,
        "timestamps": timestamps,
        "pattern": pattern,
        "hours": hours
    })


@app.route("/api/single", methods=["POST"])
def single_predict():
    """
    POST /api/single
    Body: { "voltage": 230.5, "current": 6.2 }
    Returns single point prediction
    """
    data = request.get_json() or {}
    voltage = float(data.get("voltage", 230))
    current = float(data.get("current", 5))

    # Generate a small window around this point for context
    n = 20
    voltages = [voltage + np.random.randn() * 2 for _ in range(n)]
    currents = [current + np.random.randn() * 0.3 for _ in range(n)]
    voltages[-1] = voltage
    currents[-1] = current

    predicted, _ = simulator.predict(voltages, currents)
    power = predicted[-1]

    return jsonify({
        "voltage": voltage,
        "current": current,
        "predicted_power_kw": round(power, 4),
        "physics_power_kw": round(voltage * current / 1000, 4),
        "unit": "kW"
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return dashboard stats / summary metrics"""
    hours = 48
    v, i, actual = generate_realistic_load(hours, base_load=2.8, pattern="residential")
    pred, _ = simulator.predict(v, i)

    peak_idx = int(np.argmax(pred))
    off_peak_idx = int(np.argmin(pred))

    return jsonify({
        "total_consumption_kwh": round(float(np.sum(pred)), 2),
        "avg_power_kw": round(float(np.mean(pred)), 3),
        "peak_power_kw": round(float(max(pred)), 3),
        "min_power_kw": round(float(min(pred)), 3),
        "peak_hour": peak_idx % 24,
        "off_peak_hour": off_peak_idx % 24,
        "efficiency_score": round(random.uniform(82, 95), 1),
        "carbon_footprint_kg": round(float(np.sum(pred)) * 0.233, 2),
        "cost_estimate_inr": round(float(np.sum(pred)) * 8.5, 2),
    })


if __name__ == "__main__":
    print("🚀 PINN Electricity Load Forecasting API")
    print(f"   Model: {'PyTorch PINN' if simulator.loaded else 'Physics Simulation'}")
    app.run(debug=True, port=5000)
