

# Analytica Kepler — Hunting Exoplanets with AI

[![YouTube](https://img.shields.io/badge/Watch%20Demo-YouTube-red)](https://www.youtube.com/watch?v=e_QeQttYwfk)

**Team:** Analytica Kepler (UFRJ Analytica)
**Challenge:** NASA Space Apps 2025 — *A Distant World: Hunting Exoplanets with AI*
**Stack:** Python • Jupyter • Streamlit

> We train ML/AI models on open data (Kepler/K2/TESS) and analyze **new datasets** (including K2/NEOSSat) to **prioritize/identify** exoplanet candidates. The web app supports CSV upload, automatic harmonization (via an Agent), and probability-based predictions.

---

## 🔗 Production Demo

* **App (Streamlit):** *<insert Railway URL>*
* **Loaded model:** `models/harmonized_rf_model.pkl`

---

## 🚀 What this repository provides

* **Commented notebook (PT-BR)** covering the pipeline:

  * Sample **NEOSSat FITS** download/inspection
  * Toy light curve (aperture-sum)
  * **KOI/TOI** ingestion via **NASA Exoplanet Archive (TAP)**
  * **Baseline ML** (Random Forest) for `koi_disposition`
  * **`mission` feature** (Kepler/TESS) for cross-mission robustness
* **Training script** (`src/train.py`): harmonizes KOI/TOI, maps TOI labels, trains, and saves `.pkl`
* **Agent** (`src/agent.py`) that:

  * Detects format (KOI/TOI) or uses **Gemini** to suggest mappings for unfamiliar datasets
  * Harmonizes input and applies the **model** (uses `predict_proba` and provides **insights**)
* **Streamlit app** (`app.py`) for upload → harmonization → prediction dashboard

> **Task:** from **tabular transit features** (period, duration, depth, radii), the model predicts **CONFIRMED** vs **FALSE POSITIVE** (binary baseline). “CANDIDATE/APC” are excluded from training and may appear as ambiguous during analysis.

---

## 📁 Repository structure

```
.
├─ data/               # local data (not versioned)
├─ datasets/           # small examples for testing
├─ models/             # .pkl artifacts
├─ outputs/            # reports/figures
├─ figures/            # generated images
├─ logs/               # logs
├─ notebooks/
│  └─ AnalyticaKepler_Exoplanets_Starter.ipynb
├─ src/
│  ├─ train.py
│  ├─ agent.py
│  └─ parsers.py
├─ app.py
├─ requirements.txt
├─ railway.json
└─ README.md
```

> Large files (FITS, models) are ignored via `.gitignore`, except `models/harmonized_rf_model.pkl`.

---

## ⚙️ Quickstart

```bash
# create venv
python -m venv .venv

# PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# install deps
pip install -r requirements.txt

# run locally
streamlit run app.py
```

---

## 🧪 Main notebook

1. **NEOSSat (CSA):** list/download sample **FITS**, show header & image.
2. **Toy light curve:** simple aperture-sum with background subtraction (illustrative).
3. **KOI/TOI (TAP):** HTTP (CSV). **TOI tip** — use `AS` aliases:

```sql
select top 500
  toi,
  tid as tic_id,
  tfopwg_disp,
  pl_orbper  as orbital_period,
  pl_trandep as transit_depth,
  pl_trandurh as transit_duration,
  pl_rade    as planet_radius,
  st_rad     as stellar_radius,
  st_teff    as stellar_teff
from toi
```

4. **Baseline ML:** RandomForest + `mission`.
5. **Model export:** `models/koi_rf_with_mission.pkl` (notebook) and `models/harmonized_rf_model.pkl` (script).

---

## 🧩 How we meet the challenge

* **Supervised training** (KOI/TOI) with official labels.
* **Cross-mission generalization** (Kepler/TESS; K2 handled via the Agent).
* **New data analysis:** upload → auto-harmonize → predict with **probabilities**.
* **Web interface:** Streamlit with metrics and CSV download.

> **TOI label mapping:** `CP/KP → CONFIRMED`, `FP/FA → FALSE POSITIVE`, `PC/APC → ambiguous or excluded from binary baseline`.
> **Units:** `koi_depth` in ppm → converted to fraction (`ppm/10000`).

---

## 📊 Performance metrics

* **Accuracy** (70/30 stratified, ~87%).
* **Confusion matrix** (CONFIRMED vs FALSE POSITIVE).
* **PR-AUC (CONFIRMED)** — more informative with class imbalance.

> Training saves `figures/confusion_matrix.png` and `figures/pr_curve_confirmed.png`. The app includes a **Metrics** section.

---

## 🧪 UI: hyperparameters & (re)training (near-term roadmap)

* **Hyperparam panel** (n_estimators, max_depth, class_weight)
* **Retrain with CSV** (optional):

  * user supplies **label column**
  * Agent suggests mapping → validate → (re)train and save user model
* **Adjustable threshold** for “CONFIRMED” (default 0.7)
* **CSV download** with predictions and probabilities

---

## 🛰️ K2 example (generalization)

A tiny K2 CSV sample is included in `datasets/`. The Agent harmonizes it and applies the model.

> Next: `parse_k2_data` (if needed) for an explicit schema.

---

## 🧱 Deploy (Railway)

* `railway.json` installs `requirements.txt` and runs `app.py` with `$PORT`.
* Environment variables:

  * `GOOGLE_API_KEY` (Gemini) — optional; only for AI mapping assistance.

---

## 🗣️ Pitch (1 slide)

**Problem:** manual vetting of transit candidates is time-consuming.
**Solution:** probability-based triage with an Agent that harmonizes new datasets.
**Data:** KOI/TOI (NASA), K2/NEOSSat (examples).
**Demo:** upload → harmonize → predict + PR-AUC/confusion matrix.
**Impact:** faster follow-up; less manual load.
**Roadmap:** robust NEOSSat photometry, domain adaptation, in-app retraining.

---

## 🔗 Data sources & docs

* **NASA Exoplanet Archive (TAP)**

  * Using TAP: [https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html)
  * KOI columns: [https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html](https://exoplanetarchive.ipac.caltech.edu/docs/API_kepcandidate_columns.html)
  * TOI columns: [https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)

* **NEOSSat — Canadian Space Agency (CSA)**

  * Open dataset: [https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30](https://donnees-data.asc-csa.gc.ca/en/dataset/9ae3e718-8b6d-40b7-8aa4-858f00e84b30)
  * NEOSSat page: [https://www.asc-csa.gc.ca/eng/satellites/neossat/](https://www.asc-csa.gc.ca/eng/satellites/neossat/)
  * CADC NEOSSat: [https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/neossat/](https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/neossat/)

> **License:** NEOSSat data under **Open Government Licence – Canada (OGL-Canada)**. Cite CSA/CADC and NASA Exoplanet Archive.

---

## 🤝 Contributing

* PRs and issues are welcome.
* Keep notebooks clear (PT-BR/EN comments), and provide small samples in `datasets/`.
* Avoid pushing large files (FITS/models) — use LFS or external links.

---

## 📜 Acknowledgments

* **NASA Space Apps**, **NASA Exoplanet Archive**
* **CSA** (NEOSSat) and the **open-source community** (Astropy, scikit-learn)

---

## 📣 Contact

* **Team:** Analytica Kepler — UFRJ Analytica
* **Goal:** accelerate exoplanet discovery and prioritization with open, accessible tools.

---
