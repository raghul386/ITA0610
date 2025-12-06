import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Features (like car attributes)
X = df[["Age", "Income", "Experience"]]

# Target (car price equivalent)
y = df["MonthlySpending"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict a new example (Age=30, Income=50000, Experience=5)
sample = [[30, 50000, 5]]
prediction = model.predict(sample)

print("Predicted Monthly Spending:", prediction[0])
