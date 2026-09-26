import pandas as pd

data_path = "data/coconutdb.csv"

df = pd.read_csv(data_path)

print(df.columns)