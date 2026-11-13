import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

data = pd.read_csv("housing_price_dataset.csv")

print("First 5 rows:")
print(data.head())
print("\nNumber of rows and columns:", data.shape)
print("\nData types:")
print(data.dtypes)
print("\nMissing values in each column:")
print(data.isnull().sum())
print("\nStatistics:")
print(data.describe())

# Remove missing values (if any) and duplicate rows
data = data.dropna()
data = data.drop_duplicates()

# Neighborhood is text, so convert it to numbers using one-hot encoding
data = pd.get_dummies(data, columns=["Neighborhood"], drop_first=True, dtype=int)
print("\nColumns after encoding:", list(data.columns))

# STEP 3: VISUALIZATION AND CORRELATION

plt.figure(figsize=(7, 5))
plt.scatter(data["SquareFeet"], data["Price"], alpha=0.4)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Price vs Square Feet")
plt.show()

plt.figure(figsize=(9, 6))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# STEP 4: SPLIT DATA INTO TRAIN AND TEST  (80% train, 20% test)

X = data.drop("Price", axis=1)     # all input columns
y = data["Price"]                  # target column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Helper function: prints and returns the regression metrics
def regression_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print("MAE  :", round(mae, 2))
    print("MSE  :", round(mse, 2))
    print("RMSE :", round(rmse, 2))
    print("R2   :", round(r2, 4))
    return [mae, mse, rmse, r2]

# PART A: SIMPLE LINEAR REGRESSION (only SquareFeet)

simple_model = LinearRegression()
simple_model.fit(X_train[["SquareFeet"]], y_train)
simple_pred = simple_model.predict(X_test[["SquareFeet"]])

simple_scores = regression_metrics("Simple Linear Regression", y_test, simple_pred)

# Graph: test data points with the fitted line
plt.figure(figsize=(7, 5))
plt.scatter(X_test["SquareFeet"], y_test, alpha=0.4, label="Actual")
plt.plot(X_test["SquareFeet"], simple_pred, color="red", label="Regression line")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# PART B: MULTIPLE LINEAR REGRESSION (all columns)

multi_model = LinearRegression()
multi_model.fit(X_train, y_train)
multi_pred = multi_model.predict(X_test)

multi_scores = regression_metrics("Multiple Linear Regression", y_test, multi_pred)

# Graph: actual vs predicted (points near the red line = good model)
plt.figure(figsize=(7, 5))
plt.scatter(y_test, multi_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.show()

# Coefficients show how much each feature changes the price
print("\nCoefficients of Multiple Linear Regression:")
for col, coef in zip(X.columns, multi_model.coef_):
    print(f"{col}: {coef:.2f}")

# Compare both models in one table and one graph
results = pd.DataFrame([simple_scores, multi_scores],
                       columns=["MAE", "MSE", "RMSE", "R2"],
                       index=["Simple LR", "Multiple LR"])
print("\nComparison of the two regression models:")
print(results)

results[["MAE"]].plot(kind="bar", legend=False, title="MAE Comparison (lower is better)")
plt.ylabel("MAE")
plt.xticks(rotation=0)
plt.show()


# PART C: LOGISTIC REGRESSION (classification -> gives F1-score)
# Price is converted into 2 classes: 1 = High price, 0 = Low price

median_price = y.median()
y_class = (y > median_price).astype(int)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42)

# Scale the features (logistic regression works better with scaled data)
scaler = StandardScaler()
Xc_train_s = scaler.fit_transform(Xc_train)
Xc_test_s = scaler.transform(Xc_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train_s, yc_train)
yc_pred = clf.predict(Xc_test_s)

print("\n--- Logistic Regression (High / Low price) ---")
print("Accuracy :", round(accuracy_score(yc_test, yc_pred), 4))
print("Precision:", round(precision_score(yc_test, yc_pred), 4))
print("Recall   :", round(recall_score(yc_test, yc_pred), 4))
print("F1-score :", round(f1_score(yc_test, yc_pred), 4))
print("\nClassification Report:")
print(classification_report(yc_test, yc_pred, target_names=["Low", "High"]))

# Confusion matrix graph
cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Predicted probabilities for the first 5 test houses
probs = clf.predict_proba(Xc_test_s[:5])
print("\nProbability of High price for first 5 test houses:")
print(np.round(probs[:, 1], 3))

# STEP 5: PREDICT PRICE FOR A NEW HOUSE (uses Multiple Linear Regression)

print("\n--- Predict price of a new house ---")
square_feet = float(input("Enter Square Feet: "))
bedrooms = int(input("Enter Bedrooms: "))
bathrooms = int(input("Enter Bathrooms: "))
neighborhood = input("Enter Neighborhood (Rural / Suburb / Urban): ").strip().capitalize()

new_house = pd.DataFrame(0, index=[0], columns=X.columns)
new_house["SquareFeet"] = square_feet
new_house["Bedrooms"] = bedrooms
new_house["Bathrooms"] = bathrooms
if "Neighborhood_" + neighborhood in new_house.columns:
    new_house["Neighborhood_" + neighborhood] = 1   

print("Predicted House Price:", round(multi_model.predict(new_house)[0], 2))