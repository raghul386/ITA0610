import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select features (independent variables)
X = df[["Age", "Experience"]]

# Target variable (dependent variable)
y = df["Income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Predict for a new example
sample = [[30, 5]]   # Age=30, Experience=5 yrs
prediction = model.predict(sample)

print("Predicted Income:", prediction[0])
