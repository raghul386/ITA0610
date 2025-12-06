import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Single feature
X = df[["Age"]]

# Target variable
y = df["MonthlySpending"]

# Create polynomial features (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# Build and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict for a new sample
sample = [[30]]   # Age = 30
sample_poly = poly.transform(sample)

prediction = model.predict(sample_poly)
print("Predicted Monthly Spending:", prediction[0])
