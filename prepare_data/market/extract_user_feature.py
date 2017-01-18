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
datapath = os.path.expanduser('~/storage/sa-data/market/')
data = np.genfromtxt(datapath + 'panelists12.csv',
                      delimiter='\t', missing_values='?') 




header = ['Panelist_ID', 'Panelist_Type', 'Combined_Pre-Tax_Income_of_HH', 'Family_Size', 'Household_Head_Race', 'Type_of_Residential_Possession', 'COUNTY', 'Age_Group_Applied_to_Household_Head', 'Education_Level_Reached_by_Household_Head', 'Occupation_Code_of_Household_Head', 'Age_Group_Applied_to_Male_HH', 'Education_Level_Reached_by_Male_HH', 'Occupation_Code_of_Male_HH', 'MALE_HOUR', 'MALE_SMOKE', 'Age_Group_Applied_to_Female_HH', 'Education_Level_Reached_by_Female_HH', 'Occupation_Code_of_Female_HH', 'Female_Working_Hour_Code', 'FEM_SMOKE', 'Number_of_Dogs', 'Number_of_Cats', 'Children_Group_Code', 'Marital_Status_(WARNING:_This_may_be_flipped_for_years_8-11)', 'HH_LANG', 'ALL_TVS', 'CABL_TVS', 'Hispanic_Flag', 'HISP_CAT', 'RACE2', 'RACE3', 'MICROWAVE', 'device_type', 'ZIPCODE', 'FIPSCODE', 'market_based_upon_zipcode', 'IRI_Geography_Number', 'EXT_FACT']

data[data == 99] = np.nan

keeper = ['Panelist_ID', 'Combined_Pre-Tax_Income_of_HH', 'Family_Size', 'Household_Head_Race', 'Type_of_Residential_Possession', 'Age_Group_Applied_to_Household_Head', 'Education_Level_Reached_by_Household_Head', 'Occupation_Code_of_Household_Head', 'Age_Group_Applied_to_Male_HH', 'Education_Level_Reached_by_Male_HH', 'Occupation_Code_of_Male_HH', 'MALE_HOUR', 'MALE_SMOKE', 'Age_Group_Applied_to_Female_HH', 'Education_Level_Reached_by_Female_HH', 'Occupation_Code_of_Female_HH', 'Female_Working_Hour_Code', 'FEM_SMOKE', 'Number_of_Dogs', 'Number_of_Cats', 'Children_Group_Code', 'Marital_Status', 'HH_LANG', 'ALL_TVS', 'CABL_TVS', 'Hispanic_Flag', 'HISP_CAT', 'MICROWAVE', 'device_type']


table = pandas.DataFrame(data, columns=header)

sum(table.)

print(table)
