import pandas as pd
from pandas.io import json
import seaborn as sb
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

title = sys.argv[1]
log_names = sys.argv[5:]
graph_name = sys.argv[2]
lower_limit = int(sys.argv[3])
upper_limit = int(sys.argv[4])





df = None

for log_name in log_names:
    with open(log_name) as log_file:
        log = json.load(log_file)
        temp_df = pd.DataFrame(log)
        if df is not None:
            df = df.append(temp_df)
        else:
            df = temp_df


print(df.describe())
plot = sb.lineplot(data=df, x=0, y=2, ci=None, estimator=np.median)
grouped = df.groupby(0)[2].quantile((0.25,0.75)).unstack()
plot.fill_between(x = grouped.index,y1 = grouped.iloc[:,0],y2=grouped.iloc[:,1], alpha=0.25)
plot.set_xlabel("Timestep")
plot.set_ylabel("Reward")
plot.set_title(title)
plot.set_ylim(lower_limit, upper_limit)

plot.get_figure().savefig(f"{graph_name}.png")
plot.get_figure().savefig(f"{graph_name}.svg")


