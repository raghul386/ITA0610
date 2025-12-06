import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Features
X = df[["Age", "Income", "Experience", "MonthlySpending"]]

# Target
y = df["CareerField"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------ 1. KNN ---------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_acc = accuracy_score(y_test, knn.predict(X_test))

# ------------ 2. Logistic Regression ---------------
logr = LogisticRegression(max_iter=200)
logr.fit(X_train, y_train)
logr_acc = accuracy_score(y_test, logr.predict(X_test))

# ------------ 3. Naive Bayes ---------------
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_acc = accuracy_score(y_test, nb.predict(X_test))

# ------------------- Results ----------------------
print("Accuracy Comparison:\n")
print("KNN Accuracy:", knn_acc)
print("Logistic Regression Accuracy:", logr_acc)
print("Naive Bayes Accuracy:", nb_acc)
