#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Generate a New Summary_feat dataframe 

def create_new_summary_features(df, df_Att, df_feat_summary):
    # Check values on Azdias_df to add to new summary file. 
    # Add new column 'type' and 'missing_or_unknown'
    # Drop columns 'Description', 'Value', 'Meaning'
    
    
    # Check whats new on the Dias_Attributes from AZDIAS_Feature_Summary
    df_Att = df_Att[~df_Att['Attribute'].isin(df_feat_summary.attribute)]
    df_Att = df_Att.reset_index(drop=True)
    
    
    OBJECT_VAL = lambda x: [str(k) for k in str(x).split(',')]

    # Check and map meaning for 'unknown' values on Azdias
    unknown_meaning = df_Att[df_Att['Meaning'] == 'unknown']
    unknown_meaning = unknown_meaning[unknown_meaning['Attribute'].isin(df.columns)]
    
    unknown_meaning['type'] = 'ordinal'
    unknown_meaning['missing_or_unknown'] = unknown_meaning['Value'].apply(OBJECT_VAL)

    unknown_meaning.drop(['Description', 'Value', 'Meaning'], axis=1, inplace=True)
    unknown_meaning.columns = ['attribute', 'type', 'missing_or_unknown']

    # Check and map meaning for 'no_transaction_meaning' values on Azdias
    no_transaction_meaning = df_Att[df_Att['Meaning'].isin(['no transaction known', 'no transactions known'])]
    no_transaction_meaning = no_transaction_meaning[no_transaction_meaning['Attribute'].isin(df.columns)]

    no_transaction_meaning['type'] = 'ordinal'
    no_transaction_meaning['missing_or_unknown'] = no_transaction_meaning['Value'].apply(lambda x: [str(x)])

    no_transaction_meaning.drop(['Description', 'Value', 'Meaning'], axis=1, inplace=True)
    no_transaction_meaning.columns = ['attribute', 'type', 'missing_or_unknown']

    # Check and map meaning for 'remaining_vals_meaning' values on Azdias
    remaining_vals_meaning = df_Att[~df_Att['Meaning'].isin(['unknown',
                                                             'no transaction known', 
                                                             'no transactions known'])]
    
    remaining_vals_meaning = remaining_vals_meaning[remaining_vals_meaning['Attribute'].isin(df.columns)]
    remaining_vals_meaning['type'] = remaining_vals_meaning['Meaning'].apply(lambda x: 'categorical' if x == 'Universal' else 'ordinal')
    remaining_vals_meaning['missing_or_unknown'] = remaining_vals_meaning['Value'].apply(lambda x: [])

    remaining_vals_meaning.drop(['Description', 'Value', 'Meaning'], axis=1, inplace=True)
    remaining_vals_meaning.columns = ['attribute', 'type', 'missing_or_unknown']
    
    # Update Feature_Summary
    df_feat_summary['missing_or_unknown'] = df_feat_summary['missing_or_unknown'].apply(OBJECT_VAL)
#     df_feat_summary['missing_or_unknown'] = df_feat_summary['missing_or_unknown'].apply(lambda x: x[1: -1].split(','))
    
    # Drop column
    df_feat_summary.drop(['information_level'], axis=1, inplace=True)
    summary_info = pd.concat([df_feat_summary, remaining_vals_meaning, no_transaction_meaning, unknown_meaning],sort=False, ignore_index=True)
    
    return summary_info

# feat_Info = create_new_summary_features(df, df_att, df_feat_summary )


# In[3]:


def pre_clean_df(df, feat_Info, Dias_Info):
    
    # Check on columns without Type value in Azdias_df
    cols = df.columns[~df.columns.isin(feat_Info.attribute)]
    check_cols_dtypes = df[cols].select_dtypes(include='object')
    
    describe = df[cols].describe()
    
    # Check on columns without type value in Azdias_df from Dias_info df
    dias_info = Dias_Info[Dias_Info['Attribute'].isin(cols)]
    
    cols_to_drop = list(check_cols_dtypes)
    
    last_att_info = pd.DataFrame({'attribute': list(describe.columns)})
    last_att_info['type'] = 'numeric'
    last_att_info['missing_or_unknown'] = last_att_info['type'].apply(lambda x: [])
    
    # Concat last_att_info and summary_feat_info
    summary_feat_df = pd.concat([feat_Info, last_att_info], ignore_index=True)
    
    return summary_feat_df, cols_to_drop
    
# summary_feat_df, cols_to_drop = pre_clean_df(df, feat_Info, Dias_Info) 


# In[4]:


def convert_missing_Unknown_to_Nan(df, summary_feat_df, cols_to_drop):
    # Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value
    
    for i, row in summary_feat_df.iterrows():
        name = row['attribute']
        item_features = df[name]

        if row['missing_or_unknown'] != '[]':
            items = str(row['missing_or_unknown'])[1: -1].split(',')
        else:
            items = []
        if items:
            for item in items:
                try:
                    item_features = item_features.replace([int(item)], np.NaN)
                except:
                    item_features = item_features.replace([item], np.NaN)

        df[name] = item_features
    # Drop 3 columns that are Not found in the summary_feat_df   
    df.drop(cols_to_drop, axis=1, inplace=True)
    return df

# df = convert_missing_Unknown_to_Nan(df, summary_feat_df, cols_to_drop)


# In[5]:


def plot_NaN_values(df):
    
    # Perform an assessment of how much missing data there is in each column of the
    # dataset.

    # proportion in % of NaN values
    count_nan_col = df.isna().sum() / len(df)

    # Plot the propotion
    fig = plt.figure(figsize=(12,8))
    sns.kdeplot(count_nan_col.values,shade=True, color="r")
    plt.ylabel('count of columns')
    plt.xlabel('% of NaN')
    plt.title('Distribution of missing value per columns')
    plt.show()
    


# In[7]:


# Re-Encode Features

# feat_df = summary_feat_df[summary_feat_df['attribute'].isin(Azdias_df.columns)]
# feat_df.type.value_counts()

def get_categorical_feat(df, feat_df):
    
    cat_feat = feat_df['attribute'][feat_df['type'] == 'categorical']
    # cat_feat

    binary_str_attribute = []
    binary_num_attribute = []
    multi_level_attribute = []

    for att in cat_feat:
        dtype = df[att].dtype
        count = len(df[att].value_counts())

        if count > 3:
            multi_level_attribute.append(att)
        else:
            if dtype == 'object':
                binary_str_attribute.append(att)
            else:
                binary_num_attribute.append(att)
    
    return binary_str_attribute, binary_num_attribute, multi_level_attribute
# binary_str_attribute, binary_num_attribute, multi_level_attribute = get_categorical_feat(df, feat_df)    


# In[8]:


# Re encode Multi_level_attribute column:
def encode_cat(df, feature):
    dummy_df = []
    
    for name in feature:
        dummy_df.append(pd.get_dummies(df[name], prefix=name))
    
    assert len(dummy_df) == len(feature)
    
    # drop feature from df
    df.drop(feature, axis=1, inplace=True)
    dummy_df.append(df)
    df = pd.concat(dummy_df, axis=1)
    
    return df

# df = encode_cat(df, multi_level_attribute)


# In[9]:


# Create a function that split the values in 0s and 1s
def get_wealth(x):
    try:
        x = str(x)
        return int(x[0])
    except:
        return np.NaN
    
def get_life_stage(x):
    try:
        x = str(x)
        return int(x[1])
    except:
        return np.NaN


# In[10]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

# Map Decade
dict_decade = {1: 40, 2: 40, 3: 50, 4: 50, 5: 60, 6: 60, 7: 60,8: 70, 9: 70, 10: 80, 
               11: 80, 12: 80, 13: 80, 14: 90, 15: 90, 
               np.NaN: np.NaN, -1: np.NaN, 0: np.NaN}

# Map Movement 
# Mainstream : 0, Avantgarde : 1
dict_movement = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 
                 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 
                 11: 1, 12: 0, 13: 1, 14: 0, 15: 1, 
                 np.NaN: np.NaN, -1: np.NaN, 0: np.NaN}


# In[11]:


# Investigate "LP_LEBENSPHASE_FEIN" and engineer two new variables.

# Map life_stage
life_stage = {1: 'younger_age', 2: 'middle_age', 3: 'younger_age',
              4: 'middle_age', 5: 'advanced_age', 6: 'retirement_age',
              7: 'advanced_age', 8: 'retirement_age', 9: 'middle_age',
              10: 'middle_age', 11: 'advanced_age', 12: 'retirement_age',
              13: 'advanced_age', 14: 'younger_age', 15: 'advanced_age',
              16: 'advanced_age', 17: 'middle_age', 18: 'younger_age',
              19: 'advanced_age', 20: 'advanced_age', 21: 'middle_age',
              22: 'middle_age', 23: 'middle_age', 24: 'middle_age',
              25: 'middle_age', 26: 'middle_age', 27: 'middle_age',
              28: 'middle_age', 29: 'younger_age', 30: 'younger_age',
              31: 'advanced_age', 32: 'advanced_age', 33: 'younger_age',
              34: 'younger_age', 35: 'younger_age', 36: 'advanced_age',
              37: 'advanced_age', 38: 'retirement_age', 39: 'middle_age',
              40: 'retirement_age'}

# Map fine_scale
fine_scale = {1: 'low', 2: 'low', 3: 'average', 4: 'average', 5: 'low', 6: 'low',
              7: 'average', 8: 'average', 9: 'average', 10: 'wealthy', 11: 'average',
              12: 'average', 13: 'top', 14: 'average', 15: 'low', 16: 'average',
              17: 'average', 18: 'wealthy', 19: 'wealthy', 20: 'top', 21: 'low',
              22: 'average', 23: 'wealthy', 24: 'low', 25: 'average', 26: 'average',
              27: 'average', 28: 'top', 29: 'low', 30: 'average', 31: 'low',
              32: 'average', 33: 'average', 34: 'average', 35: 'top', 36: 'average',
              37: 'average', 38: 'average', 39: 'top', 40: 'top'}


# In[12]:


# R-encode LP_life_stage and LP_fine_scale as ordinal
dict_life = {'younger_age': 1, 'middle_age': 2, 'advanced_age': 3,
            'retirement_age': 4}
dict_scale = {'low': 1, 'average': 2, 'wealthy': 3, 'top': 4}


# In[13]:


def remove_colums_above_threshold(df):
    count_nan_col = df.isna().sum() / len(df)
    
    above_30_pct = count_nan_col[count_nan_col > 0.3]
    to_drop = list(above_30_pct.index)

    df.drop(columns=to_drop, axis=1, inplace=True)
    
    return df
# num = df.shape[1]

def remove_rows_above_threshold(df, num):
    # Missing data in each row
    count_nan_row = df.isna().sum(axis = 1)
    df = df.loc[count_nan_row[count_nan_row <= 0.5 * num].index]
    df.reset_index(drop=True, inplace=True)
    return df

# df = remove_colums_above_threshold(df)
# df = remove_rows_above_threshold(df, num)


# In[14]:


def clean_data(df):
    
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
#     # Convert missing or unkown to np.nan
#     df = replace_missing_values(df)
#     df.drop(cols_to_drop, axis=1, inplace=True)
    
    # Remove Outliers
    df = remove_colums_above_threshold(df)
    num = df.shape[1]
    df = remove_rows_above_threshold(df, num)
    
    # Encoding and Engineering 
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({'OST_WEST_KZ': {'W': 1, '0': 2}}, inplace=True)
#     df = encode_cat(df, multi_level_attribute)
    
    df['PRAEGENDE_JUGENDJAHRE_Decade'] = df['PRAEGENDE_JUGENDJAHRE'].map(dict_decade)
    df['PRAEGENDE_JUGENDJAHRE_Movemnet'] = df['PRAEGENDE_JUGENDJAHRE'].map(dict_movement)
    df.drop('PRAEGENDE_JUGENDJAHRE',axis= 1, inplace=True)
    
    df['CAMEO_INTL_2015_Wealth'] = df['CAMEO_INTL_2015'].apply(lambda x: get_wealth(x))
    df['CAMEO_INTL_2015_Life_stage'] = df['CAMEO_INTL_2015'].apply(lambda x: get_life_stage(x))
    df.drop('CAMEO_INTL_2015',axis= 1, inplace=True)
    
    df['LP_life_stage'] = df['LP_LEBENSPHASE_FEIN'].map(life_stage)
    df['LP_fine_scale'] = df['LP_LEBENSPHASE_FEIN'].map(fine_scale)
    df.drop('LP_LEBENSPHASE_FEIN', axis=1, inplace=True)
    
    df['LP_life_stage'] = df['LP_life_stage'].map(dict_life)
    df['LP_fine_scale'] = df['LP_fine_scale'].map(dict_scale)
    
    df['WOHNLAGE'] = df['WOHNLAGE'].replace(0, np.nan)
    WOHNLAGE = pd.get_dummies(df['WOHNLAGE'], prefix='WOHNLAGE')
    df.drop('WOHNLAGE', axis=1, inplace=True)
    df = pd.concat([df, WOHNLAGE], axis=1)
    
    PLZ8_BAUMAX = pd.get_dummies(df['PLZ8_BAUMAX'], prefix='PLZ8_BAUMAX')
    df.drop('PLZ8_BAUMAX', axis=1, inplace=True)
    df = pd.concat([df, PLZ8_BAUMAX], axis=1)
    
    # Columns to drop
    #cols_to_Drop = ['LP_LEBENSPHASE_GROB', 'KBA05_BAUMAX']

    df.drop(columns =['LP_LEBENSPHASE_GROB', 'KBA05_BAUMAX'], axis=1, inplace=True)
    
    return df


# In[ ]:




