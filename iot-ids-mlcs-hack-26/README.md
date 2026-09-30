# MLCS-HACK-26 — IoT Intrusion Detection System

> **Disclaimer**: This project is intended for academic cybersecurity research and defensive intrusion detection using public/synthetic datasets. It does not perform live attacks, packet injection, or unauthorized network activity.

## 1. Project Overview
This project is an ML-based IoT Intrusion Detection System built for MLCS-HACK-26. It is designed to ingest cybersecurity traffic datasets (like IoT-23 or BoT-IoT), preprocess the data, extract relevant security features, and train machine learning models to detect malicious traffic.

## 2. Problem Statement
The proliferation of IoT devices has expanded the attack surface for malicious actors. Traditional signature-based detection systems often fail to identify novel attacks. This project addresses the need for intelligent, behavior-based detection of IoT intrusions using Machine Learning.

## 3. Objectives
- Automatically ingest and clean raw network traffic datasets.
- Engineer relevant cybersecurity features from network flows.
- Train and evaluate robust ML models for intrusion detection.
- Provide a clear, intuitive risk score for network traffic.
- Explain the model's predictions using Explainable AI (XAI) techniques.
- Offer an interactive, SOC-style dashboard for analysis.

## 4. Features
- **Auto-Detection**: Automatically identifies dataset structures and label columns.
- **Feature Engineering**: Dynamically computes features like `bytes_per_packet`, `packets_per_second`, and connection state flags.
- **Risk Scoring**: Maps raw prediction probabilities into actionable risk levels (LOW, MEDIUM, HIGH, CRITICAL).
- **Explainability**: Uses SHAP (when available) or built-in feature importance to explain why traffic was flagged.
- **REST API**: FastAPI backend for programmatic access to the detection engine.
- **Dashboard**: Professional Streamlit dashboard for visual analysis.

## 5. Architecture
See [docs/architecture.md](docs/architecture.md) for the full architecture diagram.

## 6. Dataset Setup
1. Create a `data/raw/` directory.
2. Place your CSV dataset (e.g., IoT-23) in this directory.
3. If no dataset is found, the system will automatically generate a **synthetic demo dataset** for testing purposes.
See [data/raw/README.md](data/raw/README.md) for detailed dataset instructions.

## 7. Installation

**Prerequisites:** Python 3.11+

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 8. Training
You can train the models using the provided CLI or batch script:

```bash
# Using Python directly
python train_model.py

# Using the batch script (Windows)
train_model.bat
```

## 9. Running Dashboard
The interactive SOC dashboard is built with Streamlit.

```bash
# Using Python directly
streamlit run app/dashboard.py

# Using the batch script (Windows)
run_dashboard.bat
```

## 10. Running API
The system provides a FastAPI backend for integration.

```bash
# Using Python directly
uvicorn app.api:app --reload

# Using the batch script (Windows)
run_api.bat
```
View the API documentation at `http://127.0.0.1:8000/docs`.

## 11. Model Explanation
The production model is a **Random Forest Classifier**, chosen for its balance of accuracy, resistance to overfitting, and interpretability in tabular data.

## 12. Feature Engineering
The `src/feature_engineering.py` module inspects the dataset and computes derived metrics if the required base columns exist. Examples:
- `bytes_per_packet`: Can indicate data exfiltration or small-packet floods.
- `src_dst_ratio`: Extreme ratios can signal scanning or one-way DoS attacks.

## 13. Evaluation Metrics
The system computes several metrics, saved in `reports/evaluation/`:
- **Detection Rate**: TP / (TP + FN) - Critical for identifying how many attacks were successfully caught.
- **FPR / FNR**: False Positive Rate and False Negative Rate.
- **F1 Score**: Harmonic mean of precision and recall.

## 14. Risk Scoring
Predictions include a Risk Score from 0-100, calculated as `attack_probability * 100`.
- **0–29**: LOW
- **30–59**: MEDIUM
- **60–79**: HIGH
- **80–100**: CRITICAL

## 15. Explainable AI
The dashboard provides a dedicated Explainability page. It uses feature importance and SHAP values to explain *why* the model made a specific prediction (e.g., "High packet rate contributed to the prediction").

## 16. Screenshots
*(Placeholder: Add screenshots of the Streamlit dashboard here during final submission)*

## 17. Limitations
- The system is designed for offline/local analysis of PCAP-derived CSV files, not inline packet inspection.
- Synthetic demo data results do not reflect real-world performance.
- Feature engineering relies on the presence of specific column names in the raw data.

## 18. Future Enhancements
- Integration with live network capturing tools (e.g., Zeek, TShark).
- Support for Deep Learning models (e.g., Autoencoders, LSTMs) for sequence analysis.
- Distributed processing using PySpark for massive datasets.

## 19. Ethical/Safety Considerations
This software is built strictly for defensive analysis and academic research. It contains no exploit code, payload generators, or active scanning mechanisms.

## 20. Team Contribution
- **Developer**: [Your Name/Team]
