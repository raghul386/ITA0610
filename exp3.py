import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("universal_dataset.csv")

# Select categorical features for ID3
df = df[["Weather", "Temperature", "Education", "City", "LoanApproved"]]

# Convert categorical values to numbers
df_encoded = df.apply(lambda col: col.astype("category").cat.codes)

# Split into X and y
X = df_encoded.drop("LoanApproved", axis=1)
y = df_encoded["LoanApproved"]

# Train Decision Tree (ID3 → criterion="entropy")
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X, y)

# Show the decision tree
plt.figure(figsize=(14,8))
plot_tree(model, 
          feature_names=X.columns, 
          class_names=["No","Yes"], 
          filled=True)
plt.show()

# Test with a new sample (example)
sample = pd.DataFrame({
    "Weather": ["Sunny"],
    "Temperature": ["Warm"],
    "Education": ["Graduate"],
    "City": ["Chennai"]
})

# Encode sample same way
sample_encoded = sample.apply(lambda col: col.astype("category").cat.codes)

# Predict
prediction = model.predict(sample_encoded)
print("Prediction (0=No, 1=Yes):", prediction[0])
