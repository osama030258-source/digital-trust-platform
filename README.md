# 🛡️ Enterprise Identity Verification Platform

An AI-powered digital identity verification platform that combines face liveness detection, deepfake detection, document OCR, and knowledge-graph-based fraud analytics into a single explainable Trust Score — with a natural-language AI Copilot for investigation support.

Built as an industry case-study project (Digital Identity, Trust & Deepfake Detection) — scoped down from an enterprise-scale spec into a working solo portfolio implementation.

---

## ✨ Features

- **Face Liveness Detection** — ResNet18-based spoof/anti-spoofing classifier (real vs. fake face)
- **Face-Match Engine** — MTCNN + InceptionResnetV1 (VGGFace2) selfie-to-ID face verification
- **Deepfake Detection** — Frame-sampled video analysis using a fine-tuned ResNet18 classifier
- **Document Intelligence (OCR)** — EasyOCR-based ID field extraction with confidence scoring
- **Trust Scoring Engine** — Weighted, explainable aggregation of all signals into a single risk score
- **Identity Knowledge Graph** — NetworkX-based fraud pattern detection (shared documents/devices across users)
- **AI Identity Copilot** — Groq (Llama 3.3 70B) powered Q&A and auto-generated investigation summaries
- **REST API** — Full FastAPI backend exposing every module as an independent endpoint
- **Executive Dashboard** — Streamlit UI for end-to-end interactive verification

---

## 🧱 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch, Torchvision |
| Face AI | facenet-pytorch (MTCNN, InceptionResnetV1) |
| OCR | EasyOCR |
| Computer Vision | OpenCV |
| Graph Analytics | NetworkX |
| LLM | Groq API (Llama 3.3 70B) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker / Docker Compose |

---

## 📂 Project Structure

```
digital-trust-platform/
├── backend/
│   ├── Dockerfile
│   └── app/
│       ├── main.py
│       ├── modules/          # face_liveness, deepfake, document_intel,
│       │                     # trust_score, knowledge_graph, copilot
│       ├── core/              # model loading
│       └── api/               # REST routes
├── ai_models/                 # trained model checkpoints (not tracked in git)
├── frontend/
│   └── dashboard.py           # Streamlit UI
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone & set up environment

```bash
git clone https://github.com/osama030258-source/digital-trust-platform.git
cd digital-trust-platform
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

### 2. Add your trained models

Place `liveness_model_best.pth` and `deepfake_model_best.pth` inside the `ai_models/` folder.

### 3. Configure environment variables

```bash
cp .env.example .env
# then add your GROQ_API_KEY
```

### 4. Run the backend (FastAPI)

```bash
uvicorn backend.app.main:app --reload --port 8000
```
Visit **http://localhost:8000/docs** for the interactive Swagger API explorer.

### 5. Run the frontend (Streamlit)

```bash
streamlit run frontend/dashboard.py
```

### 6. Or run everything with Docker

```bash
docker compose up --build
```

---

## 🔌 API Overview

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/verify/liveness` | Face liveness check |
| POST | `/api/verify/face-match` | Selfie vs ID face matching |
| POST | `/api/verify/deepfake` | Video deepfake detection |
| POST | `/api/verify/document-ocr` | Document field extraction |
| POST | `/trust-score` | Combined verification & trust score |
| POST | `/api/graph/event` | Log a verification event to the Knowledge Graph |
| GET | `/api/graph/check-document/{id}` | Detect shared-document fraud |
| POST | `/api/copilot/ask` | Ask the AI Copilot about a result |
| POST | `/api/copilot/summary` | Auto-generate an investigation summary |

Full endpoint list and schemas available at `/docs` once the server is running.

---

## 📊 Trust Scoring Model

| Component | Weight |
|---|---|
| Face Liveness | 25% |
| Face Match | 30% |
| Deepfake Check | 30% |
| Document OCR | 15% |

| Score | Risk Category |
|---|---|
| 80–100 | HIGH TRUST |
| 60–79 | MEDIUM TRUST |
| 40–59 | LOW TRUST |
| 0–39 | REJECT |

---

## 📖 Full Documentation

See [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) for datasets used, model training details, engineering challenges, and evaluation results.

---


## 🗺️ Roadmap

- [ ] Persistent Knowledge Graph (Neo4j)
- [ ] Voice authentication module
- [ ] Production-scale MRZ parsing
- [ ] Continuous/behavioral authentication

---

## 👤 Author

**Osama Khan** — BSCS (AI), University of Technology Nowshera
[GitHub](https://github.com/osama030258-source)