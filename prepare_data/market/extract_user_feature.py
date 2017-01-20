import numpy as np
import os
import cPickle as pkl
from scipy import sparse
import sys
import pandas

def save_data(savename, obsmat, obscov, index):
    reviews = dict(scores=obsmat[index, :], atts=obscov[index, :])
    pkl.dump(reviews, open(savename, 'wb'))

# load from pemb data
datapath = os.path.expanduser('../../data/market/')
data = np.genfromtxt(datapath + 'panelists12.csv',
                      delimiter='\t', missing_values='?') 




header = ['Panelist_ID', 'Panelist_Type', 'Combined_Pre-Tax_Income_of_HH', 'Family_Size', 'Household_Head_Race', 'Type_of_Residential_Possession', 'COUNTY', 'Age_Group_Applied_to_Household_Head', 'Education_Level_Reached_by_Household_Head', 'Occupation_Code_of_Household_Head', 'Age_Group_Applied_to_Male_HH', 'Education_Level_Reached_by_Male_HH', 'Occupation_Code_of_Male_HH', 'MALE_HOUR', 'MALE_SMOKE', 'Age_Group_Applied_to_Female_HH', 'Education_Level_Reached_by_Female_HH', 'Occupation_Code_of_Female_HH', 'Female_Working_Hour_Code', 'FEM_SMOKE', 'Number_of_Dogs', 'Number_of_Cats', 'Children_Group_Code', 'Marital_Status', 'HH_LANG', 'ALL_TVS', 'CABL_TVS', 'Hispanic_Flag', 'HISP_CAT', 'RACE2', 'RACE3', 'MICROWAVE', 'device_type', 'ZIPCODE', 'FIPSCODE', 'market_based_upon_zipcode', 'IRI_Geography_Number', 'EXT_FACT']

#data[data == 99] = np.nan

keeper = ['Panelist_ID', 'Combined_Pre-Tax_Income_of_HH', 'Family_Size', 'Household_Head_Race', 'Type_of_Residential_Possession', 'Age_Group_Applied_to_Male_HH', 'Education_Level_Reached_by_Male_HH', 'MALE_HOUR', 'MALE_SMOKE', 'Age_Group_Applied_to_Female_HH', 'Education_Level_Reached_by_Female_HH', 'Female_Working_Hour_Code', 'FEM_SMOKE', 'Number_of_Dogs', 'Number_of_Cats', 'Children_Group_Code', 'Marital_Status', 'HH_LANG', 'ALL_TVS', 'Hispanic_Flag', 'MICROWAVE']


table = pandas.DataFrame(data, columns=header)

for key in header:
    if key not in keeper:
        del table[key] 

table.replace(99, np.nan, True)
print(np.isnan(table).sum())

# setting missing values to the most typical one
flag = np.isnan(table['MICROWAVE'])
table.loc[flag, 'MICROWAVE'] = 0

flag = np.isnan(table['Hispanic_Flag'])
table.loc[flag, 'Hispanic_Flag'] = 2

column = table['ALL_TVS']
table.loc[np.isnan(column), 'ALL_TVS'] = 1
table.loc[column > 4, 'ALL_TVS'] = 5

flag = np.isnan(table['FEM_SMOKE'])
table.loc[flag, 'FEM_SMOKE'] = 0

flag = np.isnan(table['MALE_SMOKE'])
table.loc[flag, 'MALE_SMOKE'] = 0

flag = np.isnan(table['Combined_Pre-Tax_Income_of_HH'])
table.loc[flag, 'Combined_Pre-Tax_Income_of_HH'] = table['Combined_Pre-Tax_Income_of_HH'].mean()

key = 'Education_Level_Reached_by_Male_HH'
flag = np.isnan(table[key])
table.loc[flag, key] = table[key].mean()

key = 'Education_Level_Reached_by_Female_HH'
flag = np.isnan(table[key])
table.loc[flag, key] = table[key].mean()

key = 'MALE_HOUR'
flag = np.isnan(table[key])
table.loc[flag, key] = table[key].mean()

key = 'HH_LANG'
flag = np.isnan(table[key])
table.loc[flag, key] = 0

flag = table[key] > 1 
table.loc[flag, key] = 2

key = 'Female_Working_Hour_Code'
flag = np.isnan(table[key])
table.loc[flag, key] = table[key].mean()


key = 'Household_Head_Race'
flag = np.isnan(table[key])
table.loc[flag, key] = 1
flag = table[key] != 1
table.loc[flag, key] = 0

print(np.isnan(table).sum())

table.to_csv(datapath + 'user_feature.csv', index=False)

