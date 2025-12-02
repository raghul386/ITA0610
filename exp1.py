import csv

# Load dataset
data = []
with open("universal_dataset.csv", "r") as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        # Extract only categorical columns needed for FIND-S
        # Weather, Temperature, Education, City, LoanApproved
        filtered_row = [row[5], row[6], row[2], row[3], row[7]]
        data.append(filtered_row)

# Initialize hypothesis with '?' for each attribute
hypothesis = ["?"] * (len(data[0]) - 1)

# FIND-S algorithm
for row in data:
    attributes = row[:-1]   # X values
    label = row[-1]         # target Yes/No

    if label == "Yes":      # Only positive examples
        for i in range(len(hypothesis)):
            if hypothesis[i] == "?":
                hypothesis[i] = attributes[i]
            elif hypothesis[i] != attributes[i]:
                hypothesis[i] = "?"   # generalize

print("\nFinal Hypothesis learned by FIND-S:")
print(hypothesis)
