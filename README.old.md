# 🏛️ CivicDedupe — Intelligent Municipal Complaint Deduplication

**HackIIITD (Esya) | IIIT Delhi**

CivicDedupe is an AI-powered middleware for Indian municipalities that deduplicates multimodal citizen complaints (Image + Text + GPS) into Master Incident Tickets using spatial filtering, visual embeddings, and semantic text similarity.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Demo Scenarios (For Judges)

Use the **"Demo Quick-Fill"** dropdown in the left panel:

| Scenario | Description | Expected Result |
|----------|-------------|-----------------|
| 🔴 **A** | Sewage overflow ~18m from Ticket #1 | **DUPLICATE** → merges into Ticket #1 |
| 🟡 **B** | Broken pavement ~25m from Ticket #1 | **NEW INCIDENT** (different category) |
| 🟢 **C** | Fallen tree ~1.5km away | **NEW INCIDENT** (outside 60m radius) |

## Architecture

```
Citizen Report (Image + Text + GPS)
        │
        ▼
┌─── Spatial Pre-filter (Haversine ≤ 60m) ───┐
│                                              │
│  ┌─ Text Encoder (MiniLM-L6-v2) ──────┐    │
│  │  Cosine Similarity                  │    │
│  └─────────────────────────────────────┘    │
│  ┌─ Vision Encoder (MobileNetV3) ──────┐   │
│  │  Feature Cosine Similarity          │    │
│  └─────────────────────────────────────┘    │
│                                              │
│  Composite = 0.50×Visual + 0.35×Text        │
│            + 0.15×Spatial_Decay              │
│                                              │
│  Score ≥ 0.72 → DUPLICATE (merge)           │
│  Score < 0.72 → NEW INCIDENT                │
└──────────────────────────────────────────────┘
```

## Tech Stack

- **Frontend**: Streamlit + Folium maps + Custom CSS
- **Text AI**: sentence-transformers/all-MiniLM-L6-v2
- **Vision AI**: MobileNetV3-Small (PyTorch, CPU)
- **Database**: SQLite
- **Geo**: Haversine distance formula
