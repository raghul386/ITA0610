import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select 4 features (like Iris)
X = df[["Age", "Income", "Experience", "MonthlySpending"]]

# Multi-class target
y = df["CareerField"]

# Encode class labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# KNN model
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for new sample
sample = [[30, 50000, 5, 18000]]  
sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)
print("Predicted Career Field:", le.inverse_transform(pred)[0])
