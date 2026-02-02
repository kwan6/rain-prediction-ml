![Python](https://img.shields.io/badge/Python-3.11-blue)
![F1 Score](https://img.shields.io/badge/F1%20Score-0.80-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

# rain-prediction-ml
Leakage-aware machine learning pipeline for rain prediction using Random Forest, feature importance analysis, and F1-score evaluation on Australian weather data.

## Quick Start
1. Download dataset from Kaggle and place as `data/weather.xlsx`
2. Install deps: `pip install -r requirements.txt`
3. Train: `python src/train.py`

# About Model
Model dikembangkan untuk memprediksi kejadian hujan berdasarkan variabel meteorologi seperti suhu, kelembaban, tekanan udara, dan curah hujan historis. Algoritma dipilih karena kemampuannya menangani data non-linear dan hubungan antar fitur yang kompleks. Evaluasi menunjukkan bahwa model mampu mencapai akurasi X% dengan keseimbangan yang baik antara precision dan recall, meskipun performa masih terbatas pada kondisi cuaca ekstrem.
