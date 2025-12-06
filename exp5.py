import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select features (numeric)
X = df[["Age", "Income", "Experience"]]

# Target: CareerField
y = df["CareerField"]

# Encode target labels (IT, HR, Finance,...)
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create KNN model (k = 3)
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new sample
sample = [[30, 50000, 5]]   # Age, Income, Experience
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Predicted Career Field:", le.inverse_transform(prediction)[0])
