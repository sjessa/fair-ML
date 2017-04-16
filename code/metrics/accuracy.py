import pandas as pd
import sys
import numpy as np

data = pd.read_table(sys.argv[3], delimiter = ',')

target = sys.argv[1]  # the protected feature, and the prediction parameter
output = sys.argv[2]
 
count = data.groupby([target, output]).size()

accuracy = float(count[(0.0, 0.0)] + count[(1.0, 1.0)])/data[output].size
precision = float(count[(1.0, 1.0)])/float(count[(0.0, 1.0)] + count[(1.0, 1.0)])
recall = float(count[(1.0, 1.0)])/float(count[(1.0, 0.0)] + count[(1.0, 1.0)])
F1_score = 2*precision*recall/(precision+recall)

print "ACCURACY: " +  str(accuracy) + "\nPRECISION: " +  str(precision) + "\nRECALL: " +  str(recall) + "\nF1_SCORE: " +  str(F1_score) + "\n"