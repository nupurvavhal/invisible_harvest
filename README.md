# AI Fruit Freshness Detector

A FastAPI web app that predicts fruit freshness from an uploaded or captured image using a TensorFlow/Keras model.

## Features

- Upload an image and analyze fruit freshness
- Capture an image directly from phone/browser camera
- Detects classes like fresh/rotten apple, banana, orange, and strawberry
- Responsive web UI for desktop and mobile

## Tech Stack

- FastAPI
- Uvicorn
- TensorFlow / Keras
- Jinja2 templates
- HTML/CSS/JavaScript

## Project Structure

```text
fruit_analyzer/
  main.py
  model/
    fixedfruit_model_v2.keras
  static/
  templates/
    index.html
```

## Setup

### 1. Clone and enter the project

```powershell
git clone <your-repo-url>
cd fruit_analyzer
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install fastapi uvicorn tensorflow pillow numpy python-multipart jinja2
```

## Run the App

### Option A: Run with Python

```powershell
python main.py
```

### Option B: Run with Uvicorn (recommended for development)

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://127.0.0.1:8000

## Model File Note

This app expects the model file at:

`model/fixedfruit_model_v2.keras`


## Troubleshooting

- If PowerShell blocks venv activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

- Camera preview requires a secure context (HTTPS) on many mobile browsers.


