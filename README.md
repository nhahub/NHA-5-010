<p align="center">
  <img src="assets/MediVision - AI.png" alt="MediVision-AI Logo" width="220">
</p>

<h1 align="center">🧬 MediVision-AI</h1>

<p align="center">
  <strong>Multimodal AI Healthcare Intelligence Platform</strong><br>
  Predictive Health Analytics &bull; Computer Vision Chest X-ray Diagnostics &bull; Grad-CAM Explainability &bull; Clinical Decision Support
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Torchvision-0.17%2B-EE4C2C?style=flat" alt="Torchvision">
  <img src="https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/DenseNet-121-00599C?style=flat" alt="DenseNet-121">
  <img src="https://img.shields.io/badge/Vue.js-3.x-4FC08D?style=flat&logo=vuedotjs&logoColor=white" alt="Vue 3">
  <img src="https://img.shields.io/badge/Tailwind_CSS-3.x-38B2AC?style=flat&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
  <img src="https://img.shields.io/badge/NIH_ChestX--ray14-Dataset-red?style=flat" alt="NIH ChestX-ray14">
  <img src="https://img.shields.io/badge/CDC_NHANES-2017--2020-blue?style=flat" alt="CDC NHANES">
  <img src="https://img.shields.io/badge/Google_Gemini-Clinical_Assistant-8E75B2?style=flat&logo=google-gemini&logoColor=white" alt="Google Gemini">
  <img src="https://img.shields.io/badge/DEPI-Graduation_Project-EA580C?style=flat" alt="DEPI Graduation Project">
</p>

---

## 📌 Executive Summary

**MediVision-AI** is a clinical-grade multimodal artificial intelligence platform engineered to assist physicians, clinical researchers, and healthcare providers in proactive, data-driven patient risk assessment and radiological screening.

By uniting **Predictive Health Analytics** (evaluating 17 physiological and laboratory biomarkers from CDC NHANES surveys) with **Deep Learning Computer Vision** (classifying 5 critical thoracic pathologies on frontal chest radiographs using DenseNet-121 with **Grad-CAM visual explainability**) and a modern **Vue 3 Clinical Intelligence Dashboard**, MediVision-AI provides instant, multi-system clinical risk intelligence before acute disease progression.

---

## 🏗️ Multimodal System Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                  MediVision-AI Platform                │
                                  └───────────────────────────┬────────────────────────────┘
                                                              │
                    ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
                    │                                                                                   │
                    ▼                                                                                   ▼
┌───────────────────────────────────────┐                                           ┌───────────────────────────────────────┐
│     Modality 1: Tabular ML Engine     │                                           │     Modality 2: Computer Vision       │
│  • CDC NHANES 2017–2020 Data          │                                           │  • NIH ChestX-ray14 Radiographs       │
│  • 17 Physiological/Lab Biomarkers    │                                           │  • PyTorch DenseNet-121 Backbone      │
│  • HistGradientBoosting Classifiers   │                                           │  • Optimal Validation F1 Thresholds   │
│  • 6 Chronic Disease Targets          │                                           │  • Grad-CAM Visual Heatmap Overlays   │
└───────────────────┬───────────────────┘                                           └───────────────────┬───────────────────┘
                    │                                                                                   │
                    │ Model Artifacts (.joblib)                                                         │ Weights (.pth) & Config (.json)
                    ▼                                                                                   ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                   FastAPI Backend Service                                                 │
│   • POST /predict                          (17 Biomarkers Multi-Disease Risk Engine)                                     │
│   • POST /api/vision/predict-chest-xray     (Chest Radiograph Inference + Grad-CAM Heatmaps)                              │
│   • GET  /api/vision/status                (Vision Model Telemetry & Validation Thresholds)                              │
│   • GET  /api/vision/samples               (Built-in Radiograph Sample Assets)                                           │
│   • POST /chat                             (Bilingual Google Gemini Multimodal Clinical Assistant)                       │
│   • Mounts /assets & /app                  (Direct Browser Static Delivery)                                              │
└─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┘
                                                              │ REST API JSON & Base64
                                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           MediVision-AI Web Application (Frontend)                                        │
│  1. Modern Medical Hero Landing Page       • Dual-Engine Presentation, Interactive Navigation, Metrics Overview          │
│  2. Tabular Predictive Health Analytics    • 17 Clinical Inputs, WHO BMI Calculator, High-Risk / Healthy Presets         │
│  3. Chest X-ray Vision Diagnostics         • Drag & Drop Radiograph Upload, Sample Presets, 5 Thoracic Disease Cards     │
│  4. Grad-CAM Interpretability Studio       • Overlay / Heatmap / Original / Side-by-Side Views across all 5 Conditions   │
│  5. Floating Clinical AI Assistant         • Bilingual (EN/AR), Context-Aware Grounding, Privacy-Gated Opt-In            │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🫁 Modality 1: Computer Vision Chest X-ray Diagnostic Engine

**Status:** ✅ Active, Fully Integrated & Operational &bull; **Location:** [`models/chest_xray/`](models/chest_xray/) &bull; [`backend/src/backend/vision.py`](backend/src/backend/vision.py)

### Model Architecture & Training
- **Backbone Network:** PyTorch `torchvision.models.densenet121` initialized with ImageNet weights.
- **Classification Head:** Linear(1024 &rarr; 512) &rarr; ReLU &rarr; Dropout(0.2) &rarr; Linear(512 &rarr; 5).
- **Target Conditions (5 Thoracic Pathologies):**
  1. **Cardiomegaly** (Cardiac enlargement / cardiothoracic ratio &gt; 0.5)
  2. **Pneumonia** (Acute pulmonary parenchymal infection and consolidation)
  3. **Effusion** (Excess fluid in the pleural space / blunting of costophrenic angles)
  4. **Infiltration** (Ill-defined substance denser than air in the lung parenchyma)
  5. **Atelectasis** (Partial or complete collapse of pulmonary lung volume)
- **Input Dimensions & Preprocessing:**
  - Resize short side to `256`
  - Center crop to `224x224`
  - 3-Channel RGB normalized with ImageNet statistics: $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$

### Test Performance & Calibrated F1 Decision Thresholds
Because clinical hospital datasets exhibit natural class imbalance (e.g. Pneumonia prevalence at 1.26% vs Infiltration at 17.78%), standard 0.5 decision thresholds yield suboptimal clinical sensitivity. MediVision AI enforces **validation F1-optimal operating thresholds** derived from the validation set:

| Pathology | Prevalence (%) | Test ROC-AUC | Average Precision | Optimal F1 Threshold | Sensitivity | Specificity | F1 Score |
|:----------|:--------------:|:------------:|:-----------------:|:--------------------:|:-----------:|:-----------:|:--------:|
| **Cardiomegaly** | 2.74% | **0.918** | 0.351 | **0.939** | 39.3% | 98.4% | 0.400 |
| **Effusion** | 11.39% | **0.872** | 0.474 | **0.790** | 56.1% | 91.7% | 0.509 |
| **Atelectasis** | 10.21% | **0.799** | 0.310 | **0.650** | 47.9% | 88.5% | 0.435 |
| **Pneumonia** | 1.26% | **0.736** | 0.042 | **0.767** | 21.0% | 96.2% | 0.100 |
| **Infiltration** | 17.78% | **0.710** | 0.352 | **0.549** | 52.8% | 76.5% | 0.437 |
| **Macro Average**| — | **0.807** | — | — | — | — | — |

### Explainable AI: Grad-CAM Interpretability
- **Target Feature Map:** Hook-registered on `features.denseblock4` (the final convolutional block of DenseNet-121).
- **Activation Map Computation:** Computes global average pooled gradients $w_k^c = \frac{1}{Z}\sum_i \sum_j \frac{\partial Y^c}{\partial A_{i,j}^k}$ to produce spatial attention $L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k w_k^c A^k\right)$.
- **Visual Studio Features:**
  - **Overlay View:** Jet-colormap attention map blended with patient radiograph ($\alpha = 0.42$).
  - **Heatmap View:** Pure gradient intensity representation highlighting hot spots.
  - **Original Radiograph:** Unaltered 224x224 cropped radiograph.
  - **Side-by-Side View:** Synchronous split-screen display of original vs. Grad-CAM overlay.
  - **Interactive Condition Switching:** Clinicians can click *any* of the 5 conditions to inspect that specific pathology's convolutional activation focus without re-running inference!

---

## 📊 Modality 2: Predictive Health Analytics (Biomarker Engine)

**Status:** ✅ Active & Operational &bull; **Location:** [`models/`](models/) &bull; [`notebooks/01_MediVision_Predictive_Health_Analytics.ipynb`](notebooks/01_MediVision_Predictive_Health_Analytics.ipynb)

Analyzes individual risk across **6 chronic conditions** using 17 clinical and anthropometric biomarkers from CDC NHANES survey cohorts:

| Chronic Condition | Test ROC-AUC | National Prevalence | Clinical Significance |
|:------------------|:------------:|:-------------------:|:----------------------|
| **Diabetes** | **0.934** | ~15% | Glycemic dysregulation & insulin resistance |
| **Hypertension** | **0.811** | ~37% | Arterial pressure & cardiovascular strain |
| **Heart Disease** | **0.810** | ~8% | Coronary artery pathology & myocardial ischemia |
| **Stroke** | **0.784** | ~5% | Cerebrovascular perfusion deficits |
| **Arthritis** | **0.782** | ~31% | Degenerative joint disease & inflammation |
| **Asthma** | **0.607** | ~16% | Chronic hyper-reactive airway disease |

---

## ⚙️ Backend REST API Specification

The FastAPI microservice runs on port `8000` with comprehensive Swagger UI documentation at `http://localhost:8000/docs`.

### Vision Diagnostic Endpoints

#### 1. Analyze Chest X-ray Radiograph
- **Endpoint:** `POST /api/vision/predict-chest-xray` (alias: `POST /predict-chest-xray`)
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (*required*): Radiograph image file (PNG, JPEG, WebP, TIFF; up to 25 MB).
  - `target_condition` (*optional*): Explicit condition name to highlight in Grad-CAM.
- **Example cURL Request:**
```bash
curl -X POST "http://localhost:8000/api/vision/predict-chest-xray" \
  -F "file=@assets/samples/normal_chest_xray.png"
```
- **Example Response:**
```json
{
  "findings": [
    {
      "condition": "Cardiomegaly",
      "probability": 0.0012,
      "probability_pct": "0.1%",
      "confidence_pct": 0.1,
      "threshold": 0.9395,
      "threshold_pct": "93.9%",
      "detected": false,
      "status": "NOT DETECTED",
      "severity": "normal",
      "roc_auc": 0.918,
      "sensitivity": 0.393,
      "specificity": 0.984
    },
    {
      "condition": "Pneumonia",
      "probability": 0.6142,
      "probability_pct": "61.4%",
      "confidence_pct": 61.4,
      "threshold": 0.7666,
      "threshold_pct": "76.7%",
      "detected": false,
      "status": "NOT DETECTED",
      "severity": "elevated",
      "roc_auc": 0.736,
      "sensitivity": 0.21,
      "specificity": 0.962
    }
  ],
  "detected_conditions": [],
  "has_detected_pathology": false,
  "primary_condition": "Pneumonia",
  "summary": "No target thoracic conditions detected above validation F1 decision boundaries. All 5 evaluated conditions are within normal limits.",
  "visualizations": {
    "original": "data:image/png;base64,...",
    "overlay": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "overlays_by_condition": {
      "Cardiomegaly": "data:image/png;base64,...",
      "Pneumonia": "data:image/png;base64,..."
    },
    "heatmaps_by_condition": { ... }
  },
  "metadata": {
    "model_name": "MediVision AI Chest X-ray DenseNet-121",
    "architecture": "DenseNet-121 (Convolutional Feature Extractor)",
    "target_layer": "features.denseblock4 (Grad-CAM Localization)",
    "test_macro_roc_auc": 0.807,
    "clinical_status": "Research and development decision support only."
  }
}
```

#### 2. Get Vision Engine Status
- **Endpoint:** `GET /api/vision/status`
- **Description:** Returns model readiness, target layer configuration, and per-class metrics.

#### 3. List & Retrieve Clinical Sample Radiographs
- **Endpoint:** `GET /api/vision/samples` &rarr; Returns list of available sample images.
- **Endpoint:** `GET /api/vision/samples/{filename}` &rarr; Returns raw image file for frontend preview.

---

### Predictive Tabular Endpoints

#### Predict Chronic Disease Risks
- **Endpoint:** `POST /predict`
- **Content-Type:** `application/json`
- **Example cURL Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55, "gender": 1, "ethnicity": 3, "education": 3, "income_poverty_ratio": 1.8,
    "height_cm": 175, "weight_kg": 88, "bmi": 28.7, "waist_cm": 98,
    "hba1c": 6.2, "fasting_glucose": 115, "total_cholesterol": 220,
    "hdl": 42, "triglycerides": 180, "ldl": 142,
    "ever_smoked": 1, "sedentary_minutes": 420
  }'
```

---

### Conversational Clinical Assistant (Google Gemini)

#### Multimodal Clinical Chat
- **Endpoint:** `POST /chat`
- **Description:** Grounded, bilingual (Arabic & English) clinical explanation of both patient biomarkers and chest radiograph findings.
- **Safety Safeguard:** Requires explicit consent (`consent_to_send_health_data: true`) before analyzing profile or vision findings.

---

## 🖥️ Modern Web Application (Frontend)

The frontend is a single-page application built with **Vue.js 3**, **Tailwind CSS**, and **Font Awesome 6**.

### Highlights & Interface Views:
1. **Clinical Landing Page (`Overview`)**:
   - High-end medical hero section with gradient backdrops and pulse indicators.
   - Dual-engine architecture overview and key telemetry counters (0.934 Peak AUROC, 0.807 Vision AUROC, 17 Biomarkers, 5 Thoracic Conditions).
   - Instant "Launch Chest X-ray Vision" and "Launch Tabular Biomarker Suite" Call-to-Actions.
2. **Predictive Tabular Health Suite**:
   - 17 clinical input fields categorized by Demographics, Anthropometrics, Lab Panel, and Lifestyle.
   - Instant **High-Risk** and **Healthy** pre-calibrated patient presets.
   - Live WHO BMI classification and printable clinical report view.
3. **Chest X-ray Diagnostic Suite**:
   - Drag & drop image dropzone with real-time preview and file size validation.
   - One-click sample test buttons: **Normal Baseline** & **Pathology Case**.
   - Interactive **Grad-CAM Studio**:
     - Toggle between **Overlay**, **Heatmap**, **Original**, and **Side-by-Side** views.
     - Per-condition selector pills to inspect model focus across all 5 conditions.
     - Condition risk cards showing detection badge, probability, and optimal F1 cutoff.
4. **Floating AI Clinical Assistant**:
   - Persistent across all views with bilingual Arabic (RTL) and English support.

---

## 📂 Project Directory Structure

```text
MediVision-AI/
├── assets/
│   ├── MediVision - AI.png                    # Project branding logo
│   ├── heart_disease.png                      # Cardiovascular medical illustration
│   └── samples/
│       ├── normal_chest_xray.png              # Normal baseline radiograph sample
│       └── pathology_chest_xray.png           # Thoracic pathology radiograph sample
├── backend/
│   ├── src/
│   │   └── backend/
│   │       ├── __init__.py                    # Backend package scaffold
│   │       └── vision.py                      # DenseNet-121 inference & Grad-CAM engine
│   ├── main.py                                # FastAPI REST microservice & static router
│   ├── pyproject.toml                         # Project metadata & uv dependency spec
│   ├── requirements.txt                       # Backend service runtime dependencies
│   └── uv.lock                                # Dependency lock file
├── data/
│   ├── raw/                                   # CDC NHANES survey .XPT data
│   └── processed/                             # Processed clinical datasets
├── frontend/
│   └── index.html                             # Vue 3 + Tailwind CSS multimodal interface
├── models/
│   ├── trained_models.joblib                  # Serialized HistGradientBoosting models
│   ├── selected_features.joblib               # Pre-selected tabular feature names
│   └── chest_xray/
│       ├── chest_xray_densenet121.pth         # PyTorch DenseNet-121 model weights
│       ├── chest_xray_densenet121_metadata.json # Vision config & validation F1 thresholds
│       └── chest_xray_densenet121_metrics.json  # Evaluation metrics on test set
├── notebooks/
│   ├── 01_MediVision_Predictive_Health_Analytics.ipynb # Tabular ML training pipeline
│   ├── 02_MediVision_Chest_Xray_Vision_Pipeline.ipynb  # Chest X-ray vision pipeline (Kaggle)
│   └── NOTEBOOK_GUIDE.md                      # Guide for training notebooks
├── start_backend.bat                          # One-click Windows startup script
├── requirements.txt                           # Full project requirements
└── README.md                                  # Complete documentation
```

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.11+ (Python 3.13 recommended)
- `uv` (recommended: `pip install uv`) or standard `pip`
- Modern web browser (Chrome, Edge, Firefox, Safari)

---

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/aashour06/MediVision-AI.git
cd MediVision-AI

# Create virtual environment
uv venv
# Or with standard python: python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux/macOS:
source .venv/bin/activate
```

---

### Step 2: Install Dependencies

```bash
# Install backend and vision dependencies
uv pip install -r backend/requirements.txt
```

---

### Step 3: Configure Environment Variables (Optional)

To enable the Google Gemini clinical chat assistant, create `backend/.env`:
```bash
GEMINI_API_KEY="your-google-gemini-api-key"
```

---

### Step 4: Launch the Backend Service

**Option A — Windows Quick Launcher:**
Double-click `start_backend.bat` in the project root.

**Option B — Direct Command:**
```bash
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify service is running:
- **API Status:** `http://localhost:8000/`
- **Swagger Documentation:** `http://localhost:8000/docs`

---

### Step 5: Open the Application

You can access the full MediVision AI application through either:
1. **Direct Backend Delivery:** Navigate to `http://localhost:8000/app/` in your browser.
2. **Direct Browser Open:** Double-click `frontend/index.html` directly from your file explorer.

---

## 🔒 Privacy, Security & Ethical Safeguards

- 🔑 **No Key Exposure:** API keys are never bundled into frontend assets and remain strictly on the backend.
- 🛡️ **Explicit Patient Consent:** Health data is only shared with the Gemini assistant when the user explicitly checks the consent control.
- 🛑 **Stateless Ephemeral Processing:** Patient telemetry and radiograph images are processed strictly in-memory and are never stored or logged to disk.
- ⚠️ **Assistive Clinical Scope:** Predictions and detections are statistical estimates designed to assist licensed clinicians and are not autonomous medical diagnoses.

---

## 🎓 Academic Attribution

This software platform was developed as a Capstone Graduation Project for the **Digital Egypt Pioneers Initiative (DEPI)** under the Ministry of Communications and Information Technology (MCIT), Egypt.
