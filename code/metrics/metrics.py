import pandas as pd
import sys
import numpy as np

data = pd.read_table(sys.argv[3], delimiter = ',')

protected_feature = sys.argv[1]  # the protected feature, and the prediction parameter
output = sys.argv[2]

if len(sys.argv) > 4 and sys.argv[4] == 'False':
    favourable_outcome = 0
else:
    favourable_outcome = 1


count = pd.Series([0,0,0,0], index = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
tmp = data.groupby([protected_feature, output]).size()

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
    r_I = 'ind'
else:
    r_I = p_ypos_s1/p_ypos_s0
    
# ELIFT RATIO

if(p_ypos_s0 == 0 and p_ypos_s1 == 0):
    r_E = 'ind'
else:
    p_ypos = float(count[(0.0, favourable_outcome)] + count[(1.0, favourable_outcome)])/data[output].size
    r_E = p_ypos_s1/p_ypos

    
# ODDS RATIO
#p_yneg_s1 = float(count[(1.0, 1)])/data[output_feature].size
p_yneg_s1 = float(count[(1.0, np.abs(favourable_outcome-1))])/float(count[(1.0, np.abs(favourable_outcome-1))] + count[(1.0, favourable_outcome)])

#p_yneg_s0 = float(count[(0.0, 1)])/data[output_feature].size
p_yneg_s0 = float(count[(0.0, np.abs(favourable_outcome-1))])/float(count[(0.0, np.abs(favourable_outcome-1))] + count[(0.0, favourable_outcome)])

if(p_ypos_s0 == 0 or p_yneg_s1 == 0):
    r_OR = 'ind'
else:
    r_OR = (p_ypos_s1*p_yneg_s0)/(p_ypos_s0*p_yneg_s1)
#r_OR = 1/r_OR

print "\nIMPACT RATIO: " +  str(r_I) + "\nELIFT RATIO: " +  str(r_E) + "\nODDS RATIO: " +  str(r_OR) + "\n"
