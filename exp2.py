import csv

# Load dataset
data = []
with open("universal_dataset.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        # Weather(5), Temperature(6), Education(2), City(3), LoanApproved(7)
        data.append([row[5], row[6], row[2], row[3], row[7]])

# Number of attributes
num_attr = 4

# Initial S and G
S = ["Ø"] * num_attr
G = [["?"] * num_attr]

def is_consistent(h, x):
    for hv, xv in zip(h, x):
        if hv != "?" and hv != xv:
            return False
    return True

# Candidate-Elimination
for row in data:
    x = row[:-1]
    y = row[-1]

    # Positive example → generalize S
    if y == "Yes":
        for i in range(num_attr):
            if S[i] == "Ø":
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = "?"
        
        # Remove inconsistent hypotheses from G
        G = [g for g in G if is_consistent(g, x)]

    # Negative example → specialize G
    else:
        new_G = []
        for g in G:
            if is_consistent(g, x):     # if g wrongly covers negative example
                for i in range(num_attr):
                    if g[i] == "?":
                        new_h = g.copy()
                        new_h[i] = S[i]
                        if is_consistent(new_h, x) == False:
                            new_G.append(new_h)
            else:
                new_G.append(g)
        G = new_G

print("\nSpecific Boundary S:", S)
print("General Boundary G:", G)
