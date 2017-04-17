import numpy as np
import pandas as pd
from sklearn import preprocessing

#https://catalog.data.gov/dataset?tags=recidivism
data = pd.read_table('./3-Year_Recidivism_for_Offenders_Released_from_Prison.csv', delimiter = ',') # make sure file path is correct
data = data.drop('Offender',axis=1)

labels, levels = pd.factorize(data['Recidivism Reporting Year'])
data['Recidivism Reporting Year'] = labels

labels, levels = pd.factorize(data['Sex'])
data['Sex'] = labels

data['Hispanic'] = (data['Race - Ethnicity'].str.contains('Non'))
tmp = np.logical_not(data['Hispanic'].tolist())
data['Hispanic'] = pd.Series(tmp)
data['Hispanic'] = data['Hispanic'].astype(int)

data['White'] = (data['Race - Ethnicity'].str.contains('White'))
data = data.dropna(subset=['White'])
tmp = np.logical_not(data['White'].tolist())
data['Non-White'] = pd.Series(tmp)
data = data.dropna(subset=['Non-White'])
data['Non-White'] = data['Non-White'].astype(int)
data = data.drop('White',axis=1)



data = data.drop('Race - Ethnicity',axis=1)

data['Age At Release'] = data['Age At Release ']
data = data.drop('Age At Release ',axis=1)
data = data.replace(to_replace='Under 25', value='18-25')
data = data.dropna(subset=['Age At Release'])
labels, levels = pd.factorize(data['Age At Release'], sort=True)
data['Age At Release'] = labels

labels, levels = pd.factorize(data['Convicting Offense Classification'])
data['Convicting Offense Classification'] = labels

labels, levels = pd.factorize(data['Convicting Offense Type'])
data['Convicting Offense Type'] = labels

labels, levels = pd.factorize(data['Convicting Offense Subtype'])
data['Convicting Offense Subtype'] = labels

labels, levels = pd.factorize(data['Release Type'])
data['Release Type'] = labels

data = data.drop('Main Supervising District',axis=1)

data['Recidivism'] = (data['Recidivism - Return to Prison'].str.contains('Yes'))
data['Recidivism'] = data['Recidivism'].astype(int)
data = data.drop('Recidivism - Return to Prison',axis=1)


labels, levels = pd.factorize(data['Recidivism Type'])
data['Recidivism Type'] = labels

data['Days to Recidivism'] = data['Days to Recidivism'].fillna(value=0)


labels, levels = pd.factorize(data['New Conviction Offense Classification'])
data['New Conviction Offense Classification'] = labels

labels, levels = pd.factorize(data['New Conviction Offense Type'])
data['New Conviction Offense Type'] = labels

labels, levels = pd.factorize(data['New Conviction Offense Sub Type'])
data['New Conviction Offense Sub Type'] = labels

data['Part of Target Population'] = (data['Part of Target Population'].str.contains('Yes'))
data['Part of Target Population'] = data['Part of Target Population'].astype(int)






x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data2 = pd.DataFrame(x_scaled)
data2.columns = data.columns

data2.to_csv('./metrics/3-Year_Recidivism_for_Offenders_Released_from_Prison_PREPROCESSED.csv')




