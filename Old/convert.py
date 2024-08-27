#Adult_W5_Anon_V1.0.0.dta
import pandas as pd
data = pd.read_stata("Adult_W5_Anon_V1.0.0.dta")
data.to_csv("converted.csv", index=False)