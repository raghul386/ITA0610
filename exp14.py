import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Features (similar to house attributes)
X = df[["Age", "Experience", "MonthlySpending"]]

# Target variable (house price equivalent)
y = df["Income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict for a new sample
sample = [[30, 5, 15000]]   # Age, Experience, MonthlySpending
prediction = model.predict(sample)

print("Predicted Income (House Price Equivalent):", prediction[0])
