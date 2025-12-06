import csv
file = open("enjoysport.csv", "r")
dataset = []
reader = csv.reader(file)

for row in reader:
    dataset.append(row)
file.close()  

print("Dataset:")
for row in dataset:
    print(row)

num_attributes = len(dataset[0]) - 1

hypothesis = ["0"] * num_attributes
print("\nInitial Hypothesis:", hypothesis)

for row in dataset:
    if row[-1] == "yes":  
        for i in range(num_attributes):
            if hypothesis[i] == "0":
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = "?"

print("\nFinal Hypothesis:", hypothesis)
