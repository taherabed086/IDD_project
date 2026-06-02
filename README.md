# IDD Semantic Segmentation Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iddproject-m6hnucawhe9xjwkappkfchk.streamlit.app)

## Overview
This project implements a semantic segmentation model trained on the **India Driving Dataset (IDD)**. The application provides an interactive interface to perform real-time segmentation, classifying road elements into 10 distinct categories.

## Model Details
- **Architecture:** DeepLabV3+
- **Backbone:** ResNet-50
- **Number of Classes:** 10
- **Dataset:** India Driving Dataset

## Live Demo
Check out the live web application here: **[Live Demo on Streamlit](https://iddproject-m6hnucawhe9xjwkappkfchk.streamlit.app)**

## Project Structure
```text
IDD_project/
├── notebooks/            # Jupyter notebooks for model training and experiments
├── app.py                # Main Streamlit web application script
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files (data, model weights)
└── README.md             # Project documentation
```

## Cloud Storage (Data & Models)
> **Note:** The dataset and the trained model weights (`.pth` files) are hosted in the cloud due to their large size.
- **Dataset:** [Insert Cloud Link Here]
- **Model Weights:** [Insert Cloud Link Here]

## Local Installation & Usage
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/taherabed086/IDD_project.git
   cd IDD_project
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Weights:**
   Download the `.pth` files from the cloud link above and place them in the appropriate directory (e.g., `models/`) before running the app.

4. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
