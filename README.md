<div align="center">

# 🏠 House Price Prediction

**Predicting residential property prices in King County, Washington using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Key EDA Insights](#-key-eda-insights)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Running Locally](#-running-locally)
- [Author](#-author)

---

## 📋 Overview

This project predicts residential property prices using historical sales data from **44 cities across King County, WA**. Users input property details — bedrooms, bathrooms, square footage, condition, city, and more — and receive an instant market value estimate.

The project covers the full data science pipeline: data cleaning, exploratory data analysis, outlier treatment, feature engineering, model comparison, hyperparameter tuning, and deployment through a custom-designed Streamlit interface.

---

## ✨ Features

- 🧾 **Full property specification form** — bedrooms, bathrooms, floors, living/lot/basement area, waterfront, view quality, condition, year built, year renovated, and city (44 options)
- ⚡ **Instant prediction** on submission, no page reload
- 📊 **Model transparency** — a dedicated "About" section explaining methodology, real accuracy metrics, and model comparison
- 🎨 **Custom-designed UI** — warm, editorial visual theme built entirely with custom CSS on top of Streamlit

---

## 📊 Dataset

- **4,600 records**, 18 original columns
- **0 missing values**, 0 duplicate rows
- **Target variable:** `price`
- Covers **44 cities** across King County, Washington
- Original features include bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, year built, year renovated, city, and date of sale

### Data Cleaning & Outlier Handling
- Removed **2 records with 0 bedrooms** — found to be data entry errors, not real listings
- Removed **2 records with 0 bathrooms** using IQR and Z-score methods
- Removed **49 listings priced at $0** — identified as gifts, transfers, or non-market transactions rather than real sales
- `country` column dropped (single unique value, USA, across the entire dataset — no predictive signal)
- `street` and `statezip` dropped due to excessive unique values with limited generalizable signal

---

## 🔍 Key EDA Insights

- **Seattle** recorded the highest number of house sales — 1,460 properties
- **Clyde Hill** has the highest average house price of any city in the dataset
- Average house price across the dataset: **~$487,316**
- Price distribution is **right-skewed** — most homes cluster at lower prices, with a long tail of high-value properties
- Homes with a **basement** sell for more on average (~$528,866) than those without
- Houses with **2.5 floors** have the highest average selling price among all floor counts
- **Waterfront properties** sell for significantly more on average (~$743,219) than non-waterfront homes
- `sqft_living` has a moderate positive correlation with price (**r = 0.43**); `sqft_lot` has almost none (**r = 0.05**)
- Renovated houses sell for more on average than non-renovated ones

---

## 🔬 Methodology

<details>
<summary><strong>Click to expand full pipeline</strong></summary>
<br>

1. **Data Cleaning** — converted `date` to datetime, dropped low-signal columns (`country`, `street`, `statezip`), confirmed zero nulls/duplicates
2. **Feature Engineering** — derived `house_Age`, `is_renovated`, and `yr_since_renovated` from `yr_built`/`yr_renovated`, then dropped the original year columns once their signal was captured
3. **Outlier Treatment** — IQR and Z-score methods applied to bedrooms, bathrooms, sqft_living, sqft_lot, and price to remove invalid or non-market records
4. **Exploratory Data Analysis** — 15+ analytical questions answered on pricing drivers (city, condition, waterfront, renovation status, floor count, correlations)
5. **Preprocessing** — Yeo-Johnson power transformation applied to skewed numeric columns (`bathrooms`, `sqft_living`, `sqft_lot`, `sqft_above`); One-Hot Encoding applied to `city`
6. **Model Comparison** — Linear Regression, Random Forest, and Gradient Boosting evaluated on a held-out test split
7. **Hyperparameter Tuning** — GridSearchCV applied to Gradient Boosting (`n_estimators=300`, `learning_rate=0.1`, `max_depth=3`)
8. **Deployment** — final pipeline serialized with `joblib` and served through a Streamlit interface

</details>

---

## 🧠 Model Performance

Three models were trained and compared. **Gradient Boosting Regressor** was selected as the final model based on the best R² and lowest error.

| Model | R² Score | MAE |
|---|:---:|:---:|
| Linear Regression | 0.658 | ~$88,932 |
| Random Forest | 0.673 | ~$84,993 |
| **Gradient Boosting (tuned, selected)** | **0.694** | **~$82,291** |

<details>
<summary><strong>Full metrics for the final model</strong></summary>
<br>

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting Regressor |
| Hyperparameters | `n_estimators=300`, `learning_rate=0.1`, `max_depth=3` |
| R² Score | 0.694 |
| Mean Absolute Error | ~$82,291 |
| RMSE | ~$121,927 |
| Training samples | 3,448 |
| Features used | 14 |

</details>

**Why Gradient Boosting won:** its ability to sequentially correct residual errors gave it an edge over both the linear baseline and the bagging-based Random Forest approach — the data has non-linear relationships and feature interactions that a linear model can't capture.

### ⚠️ Known Limitations

This dataset does not include construction-quality (`grade`) or precise geolocation (`lat`/`long`) features present in the original King County housing dataset. Since location and build quality are typically the strongest predictors of house price, their absence sets a practical ceiling on achievable R² (~65–70%) regardless of model choice or tuning. This was verified by testing target encoding, log-transformation, and further hyperparameter tuning — none of which improved performance beyond this range. Treat the app's estimate as a reference point alongside comparable local listings, not a formal appraisal.

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---|---|
| Web app framework | Streamlit |
| Model training | scikit-learn (Gradient Boosting Regressor, GridSearchCV) |
| Data handling | pandas |
| Model serialization | joblib |
| Language | Python |

---

## 📂 Project Structure

```
house-price-prediction/
├── app.py                        # Streamlit app (UI + prediction logic)
├── hero.jpg                      # Hero banner image
├── house_price_model.pkl         # Trained Gradient Boosting model
├── requirements.txt              # Python dependencies
├── Project_HousePricePred.ipynb  # Full notebook: EDA, feature engineering, model training & comparison
└── README.md
```

---

## 🚀 Running Locally

<details>
<summary><strong>Click to expand setup instructions</strong></summary>
<br>

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL shown in your terminal (usually `http://localhost:8501`).

</details>

---

## 👤 Author

**Anamta Saleem**
Data Science student, Dawood University of Engineering & Technology

<div align="center">

---
Made with 🏠 and a lot of debugging

</div>
