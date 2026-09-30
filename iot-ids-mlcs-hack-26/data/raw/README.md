# Dataset Setup — MLCS-HACK-26

## Where to Obtain Datasets

### Primary: IoT-23 Dataset
- **Source:** Stratosphere IPS — Czech Technical University
- **URL:** https://www.stratosphereips.org/datasets-iot23
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0
- **Description:** Network traffic captures from IoT devices, including malicious (botnet) and benign traffic.

### Alternative: BoT-IoT Dataset
- **Source:** UNSW Sydney Cyber Range Lab
- **URL:** https://research.unsw.edu.au/projects/bot-iot-dataset
- **Description:** Simulated IoT network traffic with labelled attack categories.

## How to Use

1. Download the dataset from the official source above.
2. Extract the CSV file(s) into this directory: `data/raw/`
3. The system will auto-detect columns and labels.
4. If auto-detection fails, configure the label column in `config/config.yaml`.

## Demo Mode

If no dataset is present, the system generates a small synthetic dataset
for demonstration. Synthetic data is clearly labelled and should NOT be
used for real security conclusions.

## Important

- Do NOT download datasets from unknown or unofficial sources.
- All datasets used must be public and properly licensed.
- This project does NOT include any proprietary or restricted data.
