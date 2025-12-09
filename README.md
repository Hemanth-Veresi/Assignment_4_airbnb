# Airbnb Price Modeling – Assignment 4

This project explores how Airbnb prices vary across cities and across different tiers of cities.  
The goal is to clean the dataset, build individual city-level models, train models for groups of cities (tiers), and finally test how well a model trained on one group works on the others.

The work is organized into four parts, each in its own notebook.

---

## Project Structure

```

airbnb-bakeoff/
│
├── data/
│   ├── *.csv                         # Raw city files
│   ├── combined_clean.csv            # Cleaned dataset created in Step 1
│
├── results/
│   ├── city_results.csv              # Output from Step 2
│   ├── tier_results.csv              # Output from Step 3
│   ├── cross_tier_results.csv        # Output from Step 4
│
├── notebooks/
│   ├── 01_data_prep.ipynb
│   ├── 02_city_models.ipynb
│   ├── 03_tier_models.ipynb
│   ├── 04_cross_tier.ipynb
│
├── src/                              # (Optional) supporting python modules
│
└── README.md

```

---

#  **Step 1 – Data Preparation**

Notebook: **data_prep.ipynb**

- Load all city CSV files from the `data/` folder.
- Add a `city` column to each dataset.
- Combine them into a single dataframe.
- Keep only numeric features used for modeling.
- Convert price into numeric format.
- Save the cleaned dataset as:

```

data/combined_clean.csv

```

---

# **Step 2 – City-Level Models**

Notebook: **city_models.ipynb**

For every city with enough data:

- Train **Linear Regression**
- Train **XGBoost Regressor**
- Compute MAE, RMSE, and R²
- Save all results to:

```

results/city_results.csv

```

This part shows how well a model performs when trained only on listings from one specific city.

---

# **Step 3 – Tier-Level Models**

Notebook: **tier_models.ipynb**

Cities are grouped into three tiers:

- **Big cities** – high population
- **Medium cities**
- **Small cities**

For each tier:

- Combine data from all cities in that tier
- Train Linear Regression and XGBoost models
- Evaluate MAE, RMSE, and R²
- Save results to:

```

results/tier_results.csv

```

This helps compare whether larger and smaller cities behave differently in pricing.

---

# **Step 4 – Cross-Tier Generalization**

Notebook: **cross_tier.ipynb**

Here we test how well a model trained on one tier performs on the others:

Examples:
- Train on **small cities**, test on **medium + big**
- Train on **medium cities**, test on **small + big**
- Train on **big cities**, test on **small + medium**

We record:
- train-tier → test-tier
- model performance (MAE, RMSE, R²)

Results are saved in:

```

results/cross_tier_results.csv

```

This shows whether a model trained in one environment transfers well to another.

---

# Key Findings (based on your results)

- XGBoost consistently outperformed Linear Regression.
- Big-city models achieved the highest accuracy because larger cities have more listings and more price variety.
- Small-city models transfer reasonably well to medium cities.
- Medium-city models transfer moderately.
- Big-city models are the most stable and generalize best across tiers.

---

# How to Run the Project

1. Open the project folder in VS Code.
2. Activate your virtual environment:

```

.\venv\Scripts\activate

```

3. Install required packages:

```

pip install -r requirements.txt

```

4. Run each notebook in order:

```

01_data_prep.ipynb
02_city_models.ipynb
03_tier_models.ipynb
04_cross_tier.ipynb

```

---

# Requirements

- Python 3.10
- pandas
- numpy
- scikit-learn
- xgboost
- Jupyter Notebook

---

#  Final Notes

Each step builds on the previous one, so be sure to run them in order.  
All outputs are saved inside the `results/` folder.  
This structure keeps the workflow clean and easy to follow.
