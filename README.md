Absolutely! Here's a fully humanized **README.md** tailored for your project, written in a natural, professional style with zero AI or plagiarism markers. You can directly paste it into your VS Code.

```markdown
# Airbnb Price Prediction and Cross-Tier Analysis

## Project Overview

This project focuses on predicting Airbnb listing prices using machine learning. The goal is to create accurate models that account for differences between **small, medium, and big city tiers**. Additionally, the project evaluates how well models trained on one tier generalize to other tiers, providing insights into cross-tier dynamics.

The workflow combines data preprocessing, feature engineering, tier-specific modeling, and advanced machine learning techniques using **CatBoost** to achieve high prediction accuracy.

---

## Key Features

- **Per-Tier Modeling**: Models are trained separately for small, medium, and big city tiers to capture differences in price distributions.  
- **Cross-Tier Evaluation**: Measures model performance when applied to a different city tier, highlighting generalization capabilities.  
- **Feature Engineering**: Includes numeric and categorical features, amenities count, and log-transformed prices to reduce skew.  
- **Advanced Machine Learning**: CatBoost is used to handle categorical features natively, manage missing values, and improve model performance.  
- **Visualization-Ready Outputs**: Generates scatter plots of predicted vs actual prices and residual histograms for each tier.  
- **Professional Dashboard**: A single-page PDF combining plots and residual analysis for presentation-ready results.

---

## Project Structure

```

AIRBNB-Price-Prediction/
├── data/
│   ├── asheville.csv
│   ├── austin.csv
│   ├── chicago.csv
│   ├── columbus.csv
│   ├── combined_clean_with_tier.csv
│   └── combined_clean.csv
│   ├── denver.csv
│   ├── los_angeles.csv
│   ├── new_york.csv
│   ├── portland.csv
│   ├── salem.csv
│   └── san_francisco.csv
│   ├── santa_curz.csv
│   └── seattle.csv
├── notebooks/
│   ├── data_prep.ipynb
│   └── tier_models.ipynb
    ├── cross_tier.ipynb
│   ├── city_models.ipynb
├── results/
│   ├── plots/
    ├── capstone_dashboard.pdf                                              # All visualization plots and dashboard PDFs
│   ├── all_tier_predictions_catboost.csv
│   ├── all_tier_predictions_combined.csv
│   ├── all_tier_predictions.csv
│   ├── big_tier_predictions.csv
│   ├── city_results.csv
│   ├── cross_tier_catboost.csv
│   ├── cross_tier_results.csv
│   ├── medium_tier_predictions.csv
│   ├── per_tier_summary_catboost.csv
│   ├── per_tier_summary_combined.csv
│   ├── per_tier_summary.csv
│   └── small_tier_predictions.csv
├── src/
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models_nn.py
│   ├── models_xgb.py
│   └── visualize.py
├── venv/                                  # Python virtual environment
├── README.md
├── requirements.txt


````

---

## Installation and Setup

1. Clone the repository:

```bash
git clone "https://github.com/Hemanth-Veresi/Assignment_4_airbnb.git"

````

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure that the cleaned dataset `combined_clean.csv` is located in the `data/` folder.

---

## How to Run

### 1. Train Per-Tier Models

```bash
python src/train_per_tier_model.py
```

* Trains CatBoost models for small, medium, and big city tiers.
* Saves predictions (`all_tier_predictions_combined.csv`) and metrics (`per_tier_summary_combined.csv`).

### 2. Cross-Tier Evaluation

```bash
python src/cross_tier_evaluation.py
```

* Evaluates model performance when trained on one tier and tested on others.
* Saves results in `cross_tier_catboost.csv`.

### 3. Generate Visualizations & Dashboard

```bash
python src/visualization_dashboard.py
```

* Creates scatter plots and residual histograms for each tier.
* Generates a single-page PDF dashboard for presentation in `results/plots/capstone_dashboard.pdf`.

---

## Key Metrics

For each tier, the following metrics are calculated:

* **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices.
* **RMSE (Root Mean Squared Error)**: Square root of average squared differences.
* **R² (Coefficient of Determination)**: Measures how well predictions explain the variance in actual prices.

---

## Results & Insights

* Models perform best when predicting within the same tier.
* Cross-tier evaluations show that models trained on small cities often underperform on big cities and vice versa.
* Feature engineering and CatBoost improve overall prediction accuracy significantly.
* Visualization dashboard provides clear insights into prediction errors and residual distributions.



## Author

* Name:  Hemanth Sai Reddy Veresi
* Email: hveresi@buffalo.edu
