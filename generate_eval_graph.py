import pandas as pd
from pandas.io import json
import seaborn as sb
import sys
import json


log_names = sys.argv[1:]

df = None
print(log_names)

for log_name in log_names:
    with open(log_name) as log_file:
        log = json.load(log_file)
        temp_df = pd.DataFrame(log)
        if df is not None:
            df = df.append(temp_df)
        else:
            df= temp_df

df = df.groupby(0).agg(['mean','std','min','max','median'])
print(df)