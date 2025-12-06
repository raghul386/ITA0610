import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("universal_dataset.csv")

# Select numeric columns for clustering
X = df[["Age", "Income", "MonthlySpending"]]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create EM model (GMM) with 3 clusters
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_scaled)

# Predict cluster for each row
clusters = gmm.predict(X_scaled)

df["Cluster"] = clusters
print(df[["Age", "Income", "MonthlySpending", "Cluster"]])

# Show cluster means (in scaled units)
print("\nCluster Means (scaled):")
print(gmm.means_)

# Predict cluster for a new example
sample = [[30, 50000, 15000]]   # Age, Income, Spending
sample_scaled = scaler.transform(sample)

prediction = gmm.predict(sample_scaled)
print("\nPredicted Cluster for sample:", prediction[0])
