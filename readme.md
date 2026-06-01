# Student Performance Prediction and Causal Educational Analytics

A unified educational analytics framework combining **machine learning prediction**, **model interpretability**, **Double Machine Learning (DoubleML) causal inference**, and an interactive **Streamlit decision-support dashboard**.

## Project Overview

This repository contains the code and analysis for the paper:

**"A Unified Framework for Student Performance Prediction, Causal Effect Estimation, and Personalized Educational Intervention"**

### Main Components

- **`edu_performance_predictor.ipynb`**  
  Main analysis notebook: data exploration, preprocessing, model comparison (Ridge, RF, GB, etc.), interpretability (SHAP, Permutation Importance), and hyperparameter tuning.

- **`DoubleML.ipynb`**  
  Causal inference analysis using DoubleML to estimate Average Treatment Effects (ATE) and heterogeneous effects (CATE) for key educational interventions.

- **`student_performance_dashboard.py`**  
  Interactive Streamlit web application integrating:
  - Real-time performance prediction (Ridge Regression)
  - Causal effect visualization (DoubleML)
  - Personalized intervention recommendations (CATE)

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit dashboard
streamlit run student_performance_dashboard.py