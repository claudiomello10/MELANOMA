import pandas as pd

# Read the train.csv file
data = pd.read_csv("train.csv")

# Print the first two values
print(data.iloc[0]["image"])
