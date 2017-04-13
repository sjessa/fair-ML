
import pandas as pd
import sys
import numpy as np

data = pd.read_table('/home/adam/Desktop/discrimination_metrics/crime/crime_clean.tsv') # make sure file path is correct
#data = pd.read_table(sys.argv[3])

protected_feature = 'black' #sys.argv[1]  # the protected feature, and the prediction parameter
output_feature = 'crime' #sys.argv[2]


favourable_outcome = 0


count = data.groupby([protected_feature, output_feature]).size()
 
# IMPACT RATIO
#p_ypos_s1 = float(count[(1.0, favourable_outcome)])/data[output_feature].size
p_ypos_s1 = float(count[(1.0, favourable_outcome)])/float(count[(1.0, favourable_outcome)] + count[(1.0, np.abs(favourable_outcome-1))])


p_ypos_s0 = float(count[(0.0, favourable_outcome)])/float(count[(0.0, favourable_outcome)] + count[(0.0, np.abs(favourable_outcome-1))])
#p_ypos_s0 = float(count[(0.0, favourable_outcome)])/data[output_feature].size

r_I = p_ypos_s1/p_ypos_s0

# ELIFT RATIO
p_ypos = float(count[(0.0, favourable_outcome)] + count[(1.0, favourable_outcome)])/data[output_feature].size

r_E = p_ypos_s1/p_ypos

# ODDS RATIO
#p_yneg_s1 = float(count[(1.0, 1)])/data[output_feature].size
p_yneg_s1 = float(count[(1.0, np.abs(favourable_outcome-1))])/float(count[(1.0, np.abs(favourable_outcome-1))] + count[(1.0, favourable_outcome)])

#p_yneg_s0 = float(count[(0.0, 1)])/data[output_feature].size
p_yneg_s0 = float(count[(0.0, np.abs(favourable_outcome-1))])/float(count[(0.0, np.abs(favourable_outcome-1))] + count[(0.0, favourable_outcome)])

r_OR = (p_ypos_s1*p_yneg_s0)/(p_ypos_s0*p_yneg_s1)
#r_OR = 1/r_OR

print "\nIMPACT RATIO: " +  str(r_I) + "\nELIFT RATIO: " +  str(r_E) + "\nODDS RATIO: " +  str(r_OR) + "\n"