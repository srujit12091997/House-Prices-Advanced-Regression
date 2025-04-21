# house_price_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('data/train.csv')

# Drop irrelevant or high-missing columns (example pruning step)
df = df.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu'], errors='ignore')

# Handle missing values for simplicity (better techniques can be added)
df.fillna(df.median(numeric_only=True), inplace=True)

# Select numerical features only
df = df.select_dtypes(include=[np.number])

# ----------------------------
# 1. Separate Features & Target
# ----------------------------
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# ----------------------------
# 2. Split the Dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Scale the Features
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 4. Train the Model
# ----------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------
# 5. Make Predictions
# ----------------------------
y_pred = model.predict(X_test_scaled)

# ----------------------------
# 6. Evaluate Model
# ----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"RMSE: ${rmse:,.2f}")
print(f"R² Score: {r2:.4f}")
