
'''
NOTE: Accessing the datasets

Files are originally in .dta files for use in Stata. Using the pandas library, they have been converted into .csv files for use in python.
'''

import pandas as pd

data = pd.read_stata("Wave1/Adult_W1_Anon_V7.0.0.dta")
data.to_csv("CSV/wave1.csv", index=False)

data = pd.read_stata("Wave2/Adult_W2_Anon_V4.0.0.dta")
data.to_csv("CSV/wave2.csv", index=False)

data = pd.read_stata("Wave3/Adult_W3_Anon_V3.0.0.dta")
data.to_csv("CSV/wave3.csv", index=False)

data = pd.read_stata("Wave4/Adult_W4_Anon_V2.0.0.dta")
data.to_csv("CSV/wave4.csv", index=False)

data = pd.read_stata("Wave5/Adult_W5_Anon_V1.0.0.dta")
data.to_csv("CSV/wave5.csv", index=False)