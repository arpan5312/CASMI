"""This file encompassess all the methods required for candidate generation (isobars included) from a test dataset;
    columns needed : adduct, precursor m/z

"""


import pandas as pd
test_path = "data/test.parquet"

df = pd.read_parquet(test_path,engine="pyarrow")
pd.set_option("display.max_columns", None)

print(df)

