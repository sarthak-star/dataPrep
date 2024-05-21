import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Check and drop duplicates
def drop_duplicates(df, columns=None):
    if columns is None:
        df.drop_duplicates(inplace=True)
    else:
        df.drop_duplicates(subset=columns, inplace=True)
    return df

# Check and handle missing data
def handle_missing_data(df):
    missing_info = df.isnull().sum() / len(df)
    print("Missing data summary:")
    print(missing_info[missing_info > 0])

    numeric_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    df_cleaned = preprocessor.fit_transform(df)
    df_cleaned = pd.DataFrame(df_cleaned, columns=numeric_features.tolist() + preprocessor.transformers_[1][1]['onehot'].get_feature_names(categorical_features).tolist())

    return df_cleaned

# Function to find and handle outliers using IQR
def handle_outliers_IQR(df):
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df

# Function to apply feature scaling
def scale_features(df):
    scaler = StandardScaler()
    numeric_features = df.select_dtypes(include=['number']).columns
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df

# Save DataFrame to CSV
def save_to_csv(df, file_path='cleanedData.csv'):
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")

# Comprehensive pipeline function
def pipeline(file_path):
    df = read_data(file_path)
    df = drop_duplicates(df)
    df = handle_missing_data(df)
    df = handle_outliers_IQR(df)
    df = scale_features(df)
    save_to_csv(df)
    return df

# Main execution
if __name__ == "__main__":
    print('Hello! This is a data cleaner. Enter the file path of your file you want to clean:')
    user_file = input("Enter file path: ")
    final_df = pipeline(user_file)
    print(final_df.head())

# If you want to provide separate function calls, the users can still call them individually:

