# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load the Dataset
df = pd.read_csv('California_Housing_Dataset.csv')

# 3. Initial Exploration
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 4. Data Preprocessing

# 4.1 Handle Missing Values
print("\nChecking missing values:")
print(df.isnull().sum())

if df['total_bedrooms'].isnull().sum() > 0:
    median_bedrooms = df['total_bedrooms'].median()
    df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

# 4.2 Encode Categorical Variable
df = df.drop('ocean_proximity', axis=1)

# 4.3 Correlation Analysis
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# 4.4 Feature Selection
X = df[['median_income']]
y = df['median_house_value']

# 4.5 Feature Scaling (Optional but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Model Building
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Prediction
y_pred = model.predict(X_test)

# 8. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared Score (R²): {r2:.2f}")

# 9. Visualization
X_test_unscaled = scaler.inverse_transform(X_test)

plt.figure(figsize=(6,4))
plt.scatter(X_test_unscaled, y_test, color='blue', alpha=0.5, label='Actual Values')
plt.plot(X_test_unscaled, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Median Income vs. Median House Value')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.grid(True)
plt.show()