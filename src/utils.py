import pandas as pd

def filter_dataframe(df, pclass_filter, sex_filter, age_range):
    df_filtered = df.copy()
    
    if pclass_filter:
        df_filtered = df_filtered[df_filtered['Pclass'].isin(pclass_filter)]
    
    if sex_filter:
        df_filtered = df_filtered[df_filtered['Sex'].isin(sex_filter)]
    
    df_filtered = df_filtered[(df_filtered['Age'] >= age_range[0]) & (df_filtered['Age'] <= age_range[1])]
    
    return df_filtered

def prepare_input_data(input_data, le_sex, le_embarked, cat_imputer, num_imputer, cat_cols, num_cols):
    input_data['Sex'] = le_sex.transform(input_data['Sex'])
    input_data['Embarked'] = le_embarked.transform(input_data['Embarked'].astype(str))
    input_data[cat_cols] = cat_imputer.transform(input_data[cat_cols])
    input_data[num_cols] = num_imputer.transform(input_data[num_cols])
    
    input_processed = pd.concat([input_data[cat_cols], input_data[num_cols]], axis=1)
    return input_processed
