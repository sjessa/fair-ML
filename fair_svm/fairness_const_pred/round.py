import pandas as pd
import sys
import numpy as np

data = pd.read_table(sys.argv[1], delimiter = ',')
data['prediction'] = data['prediction'].round(decimals=0)
data.to_csv(sys.argv[1])