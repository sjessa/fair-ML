
import pandas as pd
import sys
import numpy as np

data = pd.read_table('/home/adam/Desktop/fair-ML/code/output/recidivism-massage_predictions_svm.csv', delimiter = ',') # make sure file path is correct
#data = pd.read_table(sys.argv[3])

protected_feature = 'non_white' #sys.argv[1]  # the protected feature, and the prediction parameter
output_feature = 'pred' #sys.argv[2]
target_feature = 'class'


favourable_outcome = 0

count = pd.Series([0,0,0,0], index = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
tmp = data.groupby([protected_feature, output_feature]).size()

if ((0.0, 0.0) in tmp.index): count[(0.0, 0.0)] = tmp[(0.0, 0.0)] 
if ((0.0, 1.0) in tmp.index): count[(0.0, 1.0)] = tmp[(0.0, 1.0)] 
if ((1.0, 0.0) in tmp.index): count[(1.0, 0.0)] = tmp[(1.0, 0.0)] 
if ((1.0, 1.0) in tmp.index): count[(1.0, 1.0)] = tmp[(1.0, 1.0)] 


 
# IMPACT RATIO
#p_ypos_s1 = float(count[(1.0, favourable_outcome)])/data[output_feature].size
p_ypos_s1 = float(count[(1.0, favourable_outcome)])/float(count[(1.0, favourable_outcome)] + count[(1.0, np.abs(favourable_outcome-1))])


p_ypos_s0 = float(count[(0.0, favourable_outcome)])/float(count[(0.0, favourable_outcome)] + count[(0.0, np.abs(favourable_outcome-1))])
#p_ypos_s0 = float(count[(0.0, favourable_outcome)])/data[output_feature].size

if(p_ypos_s0 == 0):
    r_I = 1.0
else:
    r_I = p_ypos_s1/p_ypos_s0
    
# ELIFT RATIO

if(p_ypos_s0 == 0 and p_ypos_s1 == 0):
    r_E = 1.0
else:
    p_ypos = float(count[(0.0, favourable_outcome)] + count[(1.0, favourable_outcome)])/data[output_feature].size
    r_E = p_ypos_s1/p_ypos

    
# ODDS RATIO
#p_yneg_s1 = float(count[(1.0, 1)])/data[output_feature].size
p_yneg_s1 = float(count[(1.0, np.abs(favourable_outcome-1))])/float(count[(1.0, np.abs(favourable_outcome-1))] + count[(1.0, favourable_outcome)])

#p_yneg_s0 = float(count[(0.0, 1)])/data[output_feature].size
p_yneg_s0 = float(count[(0.0, np.abs(favourable_outcome-1))])/float(count[(0.0, np.abs(favourable_outcome-1))] + count[(0.0, favourable_outcome)])

if(p_ypos_s0 == 0 or p_yneg_s1 == 0):
    r_OR = 1.0
else:
    r_OR = (p_ypos_s1*p_yneg_s0)/(p_ypos_s0*p_yneg_s1)
#r_OR = 1/r_OR

print "\nIMPACT RATIO: " +  str(r_I) + "\nELIFT RATIO: " +  str(r_E) + "\nODDS RATIO: " +  str(r_OR) + "\n"



count = pd.Series([0,0,0,0], index = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
tmp = data.groupby([target_feature, output_feature]).size()

if ((0.0, 0.0) in tmp.index): count[(0.0, 0.0)] = tmp[(0.0, 0.0)] 
if ((0.0, 1.0) in tmp.index): count[(0.0, 1.0)] = tmp[(0.0, 1.0)] 
if ((1.0, 0.0) in tmp.index): count[(1.0, 0.0)] = tmp[(1.0, 0.0)] 
if ((1.0, 1.0) in tmp.index): count[(1.0, 1.0)] = tmp[(1.0, 1.0)] 


accuracy = float(count[(0.0, 0.0)] + count[(1.0, 1.0)])/data[output_feature].size

if(float(count[(0.0, 1.0)] + count[(1.0, 1.0)]) == 0):
    precision = 'ind'
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