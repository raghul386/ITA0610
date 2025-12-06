import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("universal_dataset.csv")

# Select numerical columns + target
X = df[["Age", "Income", "Experience"]]
y = df["LoanApproved"]

# Encode Yes/No into 1/0
le = LabelEncoder()
y = le.fit_transform(y)

# Scale the inputs (important for ANN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Create ANN model
model = MLPClassifier(
    hidden_layer_sizes=(5,5),      # two hidden layers with 5 neurons each
    activation='relu',
    solver='adam',                 # uses backpropagation
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict a new example
sample = [[30, 50000, 5]]        # Age, Income, Experience
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
print("Prediction (0=No, 1=Yes):", prediction[0])
