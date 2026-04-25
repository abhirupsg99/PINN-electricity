# ⚡ PINN Electricity Load Forecasting — Full-Stack App

A production-grade web application for electricity load forecasting using a **Physics-Informed Neural Network (PINN)** trained on the UCI Household Power Consumption dataset.

---

## 🗂️ Project Structure

```
pinn_app/
├── backend/
│   ├── app.py              ← Flask REST API
│   ├── requirements.txt    ← Python dependencies
│   └── pinn_model.pt       ← (optional) your trained model weights
├── frontend/
│   └── index.html          ← Full frontend (zero build needed)
└── README.md
```

---

## 🚀 Quick Start

### 1. Install backend dependencies

```bash
cd backend
pip install flask flask-cors numpy scikit-learn pandas
# Optionally: pip install torch  (for real PINN inference)
```

### 2. (Optional) Save your trained model weights

In your Colab training script, after training, add:
```python
torch.save(model.state_dict(), 'pinn_model.pt')
```
Then copy `pinn_model.pt` into `pinn_app/backend/`.  
If no model file is found, the app runs in **physics-simulation mode** (P = V·I + temporal smoothing).

### 3. Start the backend

```bash
cd backend
python app.py
# → Running on http://localhost:5000
```

### 4. Open the frontend

Simply open `frontend/index.html` in your browser — no build step needed!

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check API + model status |
| GET | `/api/stats` | Dashboard KPIs (48h summary) |
| POST | `/api/single` | Single-point prediction `{voltage, current}` |
| POST | `/api/predict` | Batch prediction `{voltage:[], current:[]}` |
| POST | `/api/forecast` | Generate N-hour forecast `{pattern, hours, base_load}` |

### Example: Single Prediction
```bash
curl -X POST http://localhost:5000/api/single \
  -H "Content-Type: application/json" \
  -d '{"voltage": 230.5, "current": 6.2}'
```

### Example: Batch Forecast
```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"pattern": "residential", "hours": 24, "base_load": 2.5}'
```

---

## 🧠 PINN Architecture

```
Input: [Voltage, Current, t]  →  3 features
Layer 1: Linear(3→64) + Tanh
Layer 2: Linear(64→64) + Tanh
Layer 3: Linear(64→1)
Output: Global Active Power (kW)

Physics Loss = λ_smooth·(dP/dt)² + λ_physics·(P − V·I)² + λ_stab·(P[t]−P[t-1])²
Total Loss = MSE_data + 0.3 × Physics_Loss
```

---

## 🖥️ Frontend Features

- **Dashboard** — Live KPIs (consumption, peak, CO₂, cost), 48h forecast chart, physics residuals
- **Predict** — Single-point slider input + CSV/paste batch upload with metrics
- **Forecast** — Multi-horizon (12h–1 week) with residential/commercial/industrial patterns
- **About** — Architecture docs, API reference, training config

---

## 📊 Dataset

UCI Household Electric Power Consumption  
- 2M+ minute-level readings (2006–2010)  
- Features used: `Voltage`, `Global_intensity` → target: `Global_active_power`  
- Source: https://archive.ics.uci.edu/dataset/235

---

## 🔧 Deploying to Production

### Backend (e.g., Railway / Render)
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:$PORT
```

### Frontend
Upload `index.html` to any static host (Vercel, Netlify, GitHub Pages).  
Update the `API` constant in the script to your deployed backend URL:
```javascript
const API = "https://your-backend.railway.app/api";
```
