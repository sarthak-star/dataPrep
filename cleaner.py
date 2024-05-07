import pandas as pd
import os

# Function to read data based on file extension
def read_data(file_path):
    _ , file_ext = os.path.splitext(file_path)
    if file_ext == '.csv':
        return pd.read_csv(file_path)
    elif file_ext == '.json':
        return pd.read_json(file_path)
    elif file_ext in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unknown file format")


# Check if there are duplicates
def drop_duplicates(df, columns=None):
	if columns == None:
		df.drop_duplicates(inplace=True)
	else:
		return df.drop_duplicates(subset = columns, inplace=False)
	return df


# Check for missing values
def check_missing_data(df):
    proportion_null_rows = 100*(round(df.isnull().any(axis=1).sum()/df.any(axis=1).count(),2))
    if proportion_null_rows <= 5:
        print(f"There are {df.isnull().any(axis=1).sum()} rows with a null value. All of them are erased!")
        df.dropna(inplace=True)
    else:
        print("Too many null values, we need to check columns by columns further.")
        if df.isnull().sum().sum() > 0:
            print("\nProportion of missing values by column")
            values = 100*(round(df.isnull().sum()/df.count(),2))
            print(values)
            dealing_missing_data(df)
        else:
            print("No missing values detected!")
    return df

# handle the missing values
def dealing_missing_data(df):
    values = 100*(round(df.isnull().sum()/df.count(),2))
    to_delete = []
    to_impute = []
    to_check = []
    for name, proportion in values.items():
        if int(proportion) == 0:
            continue
        elif int(proportion) <= 10:
            to_impute.append(name)
            df.fillna(df[name].mean(), inplace=True)
        else:
            to_check.append(name)
    
    print(f"\nThe missing values in {to_impute} have been replaced by the mean.")
    print(f"The columns {to_check} should be further understood manually")


# Function to find outliers using IQR
def find_outliers_IQR(df):
    outlier_indices = []
    df = df.select_dtypes(include=['number'])
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Get the indices of outliers for feature column
        outlier_list_col = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = list(set(outlier_indices))  # Get unique indices
    return df.iloc[outlier_indices]

def save_to_csv(df, file_path='cleanedData.csv'):
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to {file_path}")


# Pipeline to run all the above tasks in a row
def pipeline(file_path):
    df1 = read_data(file_path)
    df2 = drop_duplicates(df1)
    df3 = check_missing_data(df2)
    df4 = find_outliers_IQR(df3)
    save_to_csv(df4)
    return df4

print('Hello ! This is data cleaner enter the file path of your file you want to clean')
user_file = input("Enter file path :")
final_df = pipeline(user_file)
print(final_df.head())
