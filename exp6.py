import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select ONLY categorical columns for Naive Bayes
X = df[["Weather", "Temperature", "Education", "City"]]
y = df["LoanApproved"]

# Encode all categorical columns into numbers
le_X = LabelEncoder()
for col in X.columns:
    X[col] = le_X.fit_transform(X[col])

le_y = LabelEncoder()
y = le_y.fit_transform(y)  # Yes/No → 1/0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new sample
sample = pd.DataFrame({
    "Weather": ["Sunny"],
    "Temperature": ["Warm"],
    "Education": ["Graduate"],
    "City": ["Chennai"]
})

# Encode sample
for col in sample.columns:
    sample[col] = le_X.fit_transform(sample[col])

prediction = model.predict(sample)
print("\nPrediction for new sample (0=No, 1=Yes):", prediction[0])
