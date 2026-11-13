<div align="center">

# 🏠 Housing Price Prediction using Machine Learning

**Predict the price of a house from its size, rooms, age and location, and classify it as High or Low priced.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-blue)

*Mini Project · Machine Learning (25MCA31) · III Semester MCA · Ramaiah Institute of Technology, Bengaluru*

</div>

---

## 📌 About the Project

Buying or selling a house starts with one question: **"What is it worth?"**
This project uses supervised Machine Learning to answer it. Starting from 50,000 house records, it cleans the data, explores it with graphs, trains three models, compares them with proper metrics, and finally predicts the price of a brand-new house from user input.

## ✨ Features

- 🧹 **Data cleaning**: handles missing values and duplicates
- 🔢 **One-hot encoding** of the `Neighborhood` column (Rural / Suburb / Urban)
- 📊 **Exploratory Data Analysis**: scatter plot and correlation heatmap
- 🤖 **Three models**: Simple Linear, Multiple Linear and Logistic Regression
- 📏 **Full evaluation**: MAE, MSE, RMSE, R², Accuracy, Precision, Recall, F1-score and confusion matrix
- 🔮 **Price predictor**: enter a house's details and get an estimated price

## 🗂️ Dataset

| Detail | Info |
|---|---|
| Name | Housing Price Prediction Data |
| Source | [Kaggle: Muhammad Bin Imran](https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data) |
| Size | 50,000 records (synthetic data) |
| File | `housing_price_dataset.csv` |

| Column | Type | Description |
|---|---|---|
| `SquareFeet` | Numeric | Total area of the house |
| `Bedrooms` | Numeric | Number of bedrooms |
| `Bathrooms` | Numeric | Number of bathrooms |
| `Neighborhood` | Categorical | Rural, Suburb or Urban |
| `YearBuilt` | Numeric | Year the house was built |
| `Price` | Numeric | Selling price (**target**) |

## 🔄 Workflow

```
Dataset (CSV)
     ↓
Data Cleaning + One-Hot Encoding
     ↓
Exploratory Data Analysis
     ↓
Train-Test Split (80% / 20%)
     ↓
┌──────────────────┬────────────────────┬──────────────────────┐
│ Simple Linear    │ Multiple Linear    │ Logistic Regression  │
│ (SquareFeet)     │ (all features)     │ (High / Low price)   │
└──────────────────┴────────────────────┴──────────────────────┘
     ↓
Evaluation  →  Predict price of a new house
```

## 📈 Results

All results are measured on the 20% test set.

### Regression (predicting the price)

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Simple Linear Regression (SquareFeet only) | 39,675 | 49,662 | 0.5703 |
| **Multiple Linear Regression (all features)** | **39,430** | **49,358** | **0.5756** |

### Classification (High or Low price)

| Accuracy | Precision | Recall | F1-score |
|---|---|---|---|
| 0.8054 | 0.8060 | 0.8037 | 0.8049 |

### 💡 Key Findings

- **SquareFeet is the most important factor.** It has a correlation of about **0.75** with price, while bedrooms and bathrooms show only a very weak relationship.
- Multiple Linear Regression is the best regression model, but only slightly better than the simple one, because house size already explains most of the price.
- The Logistic Regression classifier correctly labels about **80.5%** of houses as High or Low priced.

## 🖼️ Screenshots

<p align="center">
  <img src="images/scatter_plot.png" width="48%" alt="Price vs Square Feet">
  <img src="images/heatmap.png" width="48%" alt="Correlation heatmap">
</p>
<p align="center">
  <img src="images/actual_vs_predicted.png" width="48%" alt="Actual vs predicted prices">
</p>

## 🛠️ Tech Stack

| Purpose | Tools |
|---|---|
| Language | Python |
| Data handling | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn |

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Install the required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the project**
```bash
# If you used a Python script
python main.py

# If you used a Jupyter notebook
jupyter notebook
```

**4. Predict a new house price:** when the prediction part runs, enter the square feet, bedrooms, bathrooms and neighborhood (Rural / Suburb / Urban) and the model returns the estimated price.

## 📁 Project Structure

```
├── housing_price_dataset.csv     # Dataset
├── main.py / notebook.ipynb      # Complete code
├── images/                       # Graphs used in this README
├── ML_Mini_Project_Report.docx   # Detailed project report
└── README.md
```

## ⚠️ Limitations

- Linear models cannot capture non-linear patterns.
- The dataset is synthetic with only a few features. Real prices also depend on exact location, amenities and market trends.
- The High / Low label is created artificially using the median price.
- Only one train-test split was used, with no cross-validation or hyperparameter tuning.

## 🔭 Future Scope

- Add Ridge / Lasso, Decision Tree, Random Forest and Gradient Boosting models
- Use k-fold cross-validation and polynomial features
- Try a real-world housing dataset with more features
- Build a simple web app for price prediction (for example with Streamlit)

## 📚 References

1. M. B. Imran, "Housing Price Prediction Data," Kaggle.
2. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
3. scikit-learn documentation: https://scikit-learn.org/stable/
4. W. McKinney, "Data Structures for Statistical Computing in Python," 2010 (pandas).

## 👩‍💻 Author

**Shreya Gupta**
MCA Student, Ramaiah Institute of Technology, Bengaluru

---

<div align="center">

⭐ If you found this project useful, please give it a star!

</div>
