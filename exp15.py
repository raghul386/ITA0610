import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select 4 features (Iris uses 4 too)
X = df[["Age", "Income", "Experience", "MonthlySpending"]]

# Target: multi-class
y = df["CareerField"]

# Encode labels (IT, HR, Finance, Marketing)
le = LabelEncoder()
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Naive Bayes Model
model = GaussianNB()

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict for new sample
sample = [[30, 50000, 5, 18000]]
prediction = model.predict(sample)

print("Predicted Career Field:", le.inverse_transform(prediction)[0])
