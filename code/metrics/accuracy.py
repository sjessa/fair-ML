import pandas as pd
import sys
import numpy as np

data = pd.read_table(sys.argv[3], delimiter = ',')

target = sys.argv[1]  # the protected feature, and the prediction parameter
output = sys.argv[2]
 
count = pd.Series([0,0,0,0], index = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
tmp = data.groupby([target, output]).size()

if ((0.0, 0.0) in tmp.index): count[(0.0, 0.0)] = tmp[(0.0, 0.0)] 
if ((0.0, 1.0) in tmp.index): count[(0.0, 1.0)] = tmp[(0.0, 1.0)] 
if ((1.0, 0.0) in tmp.index): count[(1.0, 0.0)] = tmp[(1.0, 0.0)] 
if ((1.0, 1.0) in tmp.index): count[(1.0, 1.0)] = tmp[(1.0, 1.0)] 


accuracy = float(count[(0.0, 0.0)] + count[(1.0, 1.0)])/data[output].size

if(float(count[(0.0, 1.0)] + count[(1.0, 1.0)]) == 0):
    precision = 1.0
else:
    precision = float(count[(1.0, 1.0)])/float(count[(0.0, 1.0)] + count[(1.0, 1.0)])


if(float(count[(1.0, 0.0)] + count[(1.0, 1.0)]) == 0):
    recall = 'ind'
else:
    recall = float(count[(1.0, 1.0)])/float(count[(1.0, 0.0)] + count[(1.0, 1.0)])

if(precision == 'ind' or recall == 'ind'):
    F1_score = 'ind'
else:
    F1_score = 2*precision*recall/(precision+recall)
    
print "ACCURACY: " +  str(accuracy) + "\nPRECISION: " +  str(precision) + "\nRECALL: " +  str(recall) + "\nF1_SCORE: " +  str(F1_score) + "\n"