# SkinSafe AI — System Setup Guide

A simple, step-by-step guide to set up and run the entire SkinSafe AI platform on any new computer (Windows, macOS, or Linux).

---

## Prerequisites

Before starting, ensure the following are installed:

1. **Python 3.11** — [Download Python](https://www.python.org/downloads/)
2. **Node.js 18+ & npm** — [Download Node.js](https://nodejs.org/)
3. **MongoDB Community Server** — [Download MongoDB](https://www.mongodb.com/try/download/community) *(Make sure MongoDB service is running)*

---

## Quick Setup (In 4 Steps)

### Step 1: Create Environment Configuration (`.env`)

In the project root folder, copy `.env.example` to create `.env`:

**Windows (PowerShell):**

```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**

```bash
cp .env.example .env
```

---

### Step 2: Set Up Python Backend Environment

1. **Create Virtual Environment with Python 3.11:**

   * **Windows:**

     ```powershell
     py -3.11 -m venv .venv
     ```

     *(or `python -m venv .venv` if Python 3.11 is your default)*
   * **Linux / macOS:**

     ```bash
     python3.11 -m venv .venv
     ```
2. **Activate Virtual Environment:**

   * **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     source .venv/bin/activate
     ```
3. **Install Dependencies:**

   * **For NVIDIA GPU Acceleration (e.g. RTX 3050 / RTX 40-series with CUDA):**
     ```powershell
     pip install --upgrade pip
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
     pip install -r requirements.txt
     ```
   * **For CPU-Only (No NVIDIA GPU):**
     ```powershell
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
4. **Verify GPU / CUDA Acceleration (Optional):**

   ```powershell
   python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode')"
   ```

---

### Step 3: Set Up React Frontend

Open a new terminal in the `frontend/` folder:

```bash
cd frontend
npm install
```

---

## Running the Application

### 1. Start the Backend API Server (Port 5000)

From the project root directory (with `.venv` activated):

```bash
python run_server.py
```

* Backend will be live at: **`http://127.0.0.1:5000`**

---

### 2. Start the Frontend Web App (Port 5173)

From the `frontend/` directory:

```bash
npm run dev
```

* Open your browser at: **`http://localhost:5173`**

---

## Verification & Useful Commands

### 1. Run Automated Unit Tests (22 Tests)

```bash
pytest -v
```

### 2. View Research Benchmark Performance Table

```bash
python view_table.py
```

### 3. Verify Dataset Structure

```bash
python ml/src/organize_dataset_folders.py
```

### 4. Train the Model (Optional)

* **Dry Run (Quick Pipeline Check):**
  ```bash
  python ml/src/train.py --dry-run
  ```
* **Full Production Training (30 Epochs):**
  ```bash
  python ml/src/train.py
  ```

---

## Project Directory Structure

```
major_project/
├── backend/            # Flask REST API & MongoDB Authentication
├── frontend/           # React + Vite Medical Dashboard UI
├── ml/                 # PyTorch EfficientNet-B0 Model, Checkpoints & Reports
├── dataset/            # ISIC 2019 Dataset (organized_by_split & splits CSVs)
├── storage/            # Runtime uploads & Grad-CAM visual heatmaps
├── run_server.py       # Backend server entry point
├── view_table.py       # Terminal CLI table viewer
└── SETUP.md            # System setup guide
```

---

## Direct Execution Commands (Windows)

```powershell
.\.venv\Scripts\python.exe run_server.py 
 npm run dev
.\.venv\Scripts\python.exe view_table.py
.\.venv\Scripts\python.exe .\inspect_model.py
```
