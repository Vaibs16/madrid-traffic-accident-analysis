# Madrid Traffic Accident Analysis (2025)
Severity Prediction · Alcohol Detection · Black Spot Mapping

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF)

## Overview
Data Science project analyzing 41,000+ traffic accidents recorded by the Madrid City Council in 2025.
The goal: show that data can shift emergency response from reactive to preventive.

> The notebook is written in Spanish (university project).

## Three Problems, Three Approaches

| Model | Goal | Technique | Result |
|-------|------|-----------|--------|
| **A – Severity** | Predict if an accident requires hospitalization | XGBoost + threshold tuning | Detects 55% of serious accidents |
| **B – Alcohol** | Identify when and where to run sobriety checks | XGBoost + SHAP | 00–06h window has 7x higher risk |
| **C – Hotspots** | Find high-concentration accident zones | K-Means clustering | 5 risk zones independent of city districts |

## Key Techniques
- Feature engineering: hour extraction, weekend flag, UTM to GPS coordinate conversion
- Class imbalance handling: `scale_pos_weight` (18:1 ratio)
- Decision threshold optimization via precision-recall curve
- Model explainability with SHAP values
- Interactive map with Folium (OpenStreetMap)

## Main Finding
Individual variables (hour, location, alcohol) show near-zero correlation with accident severity.
The problem is non-linear: severity only emerges from combinations of factors, which is why
three different algorithms (XGBoost, Random Forest, Bagging) all converge at the same 35% recall ceiling, 
not a model limitation, but a data one.

## How to Run
1. Download the dataset (see `data/README.md`)
2. Open `madrid-traffic-accident-analysis.ipynb`
3. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap folium pyproj
```

## References
- Chen & Guestrin (2016) – XGBoost
- Lundberg & Lee (2017) – SHAP Values
- Lloyd (1982) – K-Means
- Dataset: Accidentalidad Madrid 2025, Ayuntamiento de Madrid
