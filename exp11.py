import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# ------------------------------
# Create Credit Score Label
# ------------------------------
credit_score = []

for i in range(len(df)):
    income = df.loc[i, "Income"]
    exp = df.loc[i, "Experience"]
    
    if income > 60000 and exp > 5:
        credit_score.append("Good")
    elif income >= 40000:
        credit_score.append("Average")
    else:
        credit_score.append("Bad")

df["CreditScore"] = credit_score

# ------------------------------
# Prepare features & target
# ------------------------------
X = df[["Age", "Income", "Experience"]]
y = df["CreditScore"]

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ------------------------------
# Train KNN Classifier
# ------------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict credit score for a new example
sample = [[30, 50000, 5]]  # age, income, experience
sample_scaled = scaler.transform(sample)

pred = model.predict(sample_scaled)
print("Predicted Credit Score:", le.inverse_transform(pred)[0])
