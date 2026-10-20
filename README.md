<div align="center">

<img src="assets/voxel-education-banner.svg" alt="Voxel Education Cost Predictor" width="100%">


# 🎓 International Education Cost Predictor

**Estimate the cost of studying abroad — with a little emerald-powered UI.**

[![Python](https://img.shields.io/badge/Python-3.x-245b32?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-245b32?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-245b32?style=for-the-badge)](https://scikit-learn.org/)

[**Source repository ↗**](https://github.com/dARSHANdR4/International-Education-Cost-Predictor) · [**Developer profile ↗**](https://github.com/dARSHANdR4)

</div>

<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

## 🌍 Project overview

A Streamlit application that estimates the total cost of pursuing higher education abroad. Choose a country and degree level, adjust study duration and living-cost assumptions, and review an estimated cost breakdown. Results can be displayed in USD or INR.

> **Hosted demo:** A verified public deployment URL is not currently configured in this repository. Follow the local setup below to run the app.

## ✨ Features

- **Cost estimate:** Predicts total education cost from country, degree level, course duration, living-cost index, rent, insurance, and visa fee inputs.
- **USD / INR display:** Shows estimates in US dollars and optionally Indian rupees.
- **Cost breakdown:** Visualizes tuition, living expenses, and visa fees.
- **Country comparison:** Compares estimates with data averages for selected countries and programs.
- **Model persistence:** Loads local model/preprocessor files when present and otherwise attempts to download the dataset and train a model.

## 🧠 Model and data

The app uses a scikit-learn pipeline with a `RandomForestRegressor`, one-hot encoding for categorical inputs, and scaling for numeric features. During training, the dataset is split into training and holdout sets and R² / RMSE are displayed.

Dataset source: [International Education Costs CSV](https://hebbkx1anhila5yf.public.blob.vercel-storage.com/International_Education_Costs-LeyxyBrErfLztS2WwEbWNzwX0Qvt8g.csv).

The app derives an estimated total as:

`tuition + (annual living cost × duration) + visa fee`

These are model-based estimates from the supplied dataset, not official university quotes or guaranteed future costs.

## 🧰 Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · Plotly · Matplotlib · Seaborn · Requests

## 🚀 Run locally

### 1. Clone the repository

```bash
git clone https://github.com/dARSHANdR4/International-Education-Cost-Predictor.git
cd International-Education-Cost-Predictor
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements and start Streamlit

```bash
python -m pip install -r requirements.txt
streamlit run education_cost_predictor.py
```

On first run, the app may need internet access to fetch its public CSV dataset and train the model. It also creates local pickle files for the model and preprocessing/data objects.

## ⚠️ Important limitations

- The current source uses a **hard-coded USD→INR conversion rate of ₹85.42 per USD**. Treat INR output as illustrative until the rate is updated or replaced with a verified live source.
- Model quality depends on the data and feature coverage; an in-app R² / RMSE score is not independent real-world validation.
- Estimates should be checked against current tuition, rent, insurance, visa fees, scholarship details, and official university information before making financial decisions.

## 🗺️ Next quests

- Add a verified hosted deployment and health/availability note.
- Replace the hard-coded currency rate with a reliable, date-stamped source.
- Add automated smoke tests and a reproducible evaluation report.

<p align="center"><img src="https://raw.githubusercontent.com/dARSHANdR4/dARSHANdR4/main/assets/minecraft-emerald-divider.svg?v=4" width="100%" alt="Animated emerald pixel divider"></p>

<div align="center">

**Built by [dARSHANdR4](https://github.com/dARSHANdR4) · Voxel Builder / Developer / Automation**

</div>


## Copyright and permissions

Copyright © 2026 dARSHANdR4. All rights reserved.

Unless otherwise stated, original project code and assets are reserved and no open-source licence is granted. Third-party materials remain subject to their respective licences. See [COPYRIGHT](./COPYRIGHT) for the full notice.
