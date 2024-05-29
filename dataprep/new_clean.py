
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

# Function to read data based on file extension
def read_data(file_path):
    _, file_ext = os.path.splitext(file_path)
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.json':
        return pd.read_json(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unknown file format")
    
def drop_duplicates(df, columns=None):
    if columns is None:
        df.drop_duplicates(inplace=True)
    else:
        df.drop_duplicates(subset=columns, inplace=True)
    return df


def handle_missing_values(df):
    # Numeric columns: Fill missing values with the mean of the column
    numeric_features = df.select_dtypes(include=['number']).columns
    
    for col in numeric_features:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        col_range = df[col].max() - df[col].min()
        # If the range is above a certain threshold, scale the column
        #if col_range > 10:  # Adjust this threshold based on your data
        #    scaler = StandardScaler()
        #    df[col] = scaler.fit_transform(df[[col]])
    
    # String columns: Fill missing values with the most frequent value (mode) of the column
    string_features = df.select_dtypes(include=['object']).columns
    for col in string_features:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
    
    return df

def label_encode_dataframe(df):
    label_encoders = {}
    mappings = {}
    
    # Iterate through each column
    for column in df.select_dtypes(include=['object']).columns:
        # Initialize LabelEncoder
        if df[column].nunique() > 25:
            print(f"Column '{column}' has too many unique values ({df[column].nunique()}). One-hot encoding is not applied.")
            continue  # Skip one-hot encoding for this column
        le = LabelEncoder()
        
        # Fit and transform the column
        df[column] = le.fit_transform(df[column].astype(str))
        
        # Store the encoder
        label_encoders[column] = le
        
        # Create a dictionary for the mapping
        mappings[column] = dict(zip(le.classes_, le.transform(le.classes_)))
    
    # Print the mappings
    for column, mapping in mappings.items():
        print(f"Mapping for column '{column}':")
        for k, v in mapping.items():
            print(f"  {k} -> {v}")
            
    
    return df

def handle_outliers_IQR(df):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df

def remove_highly_correlated_features(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop)

def save_to_csv(df, file_path='cleanedDataFile.csv'):
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

def cleaner(df):
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = label_encode_dataframe(df)
    df = handle_outliers_IQR(df)
    df = remove_highly_correlated_features(df)
    save_to_csv(df)
    return df
    
     



if __name__ == "__main__":
    print('Hello! This is a data preprocessor to automate your routine data pre processing tasks. Enter the file path of your file you want to clean:')
    user_file = input("Enter file path: ")
    df= read_data(user_file)
    print(df.head())
    df= cleaner(df)
    print("Cleaned data is saved in cleanedData.csv in the current directory")
    print(df.head())
    
