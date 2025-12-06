import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select numeric features
X = df[["Age", "Income", "Experience"]]

# Target column
y = df["LoanApproved"]

# Encode Yes/No to 1/0
le = LabelEncoder()
y = le.fit_transform(y)

# Scale numeric features (important for LR)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new example
sample = [[30, 50000, 5]]   # Age, Income, Experience
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Prediction (0=No, 1=Yes):", prediction[0])
