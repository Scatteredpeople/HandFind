import pandas as pd

# Secondary cleaning and label reconstruction for gesture recognition dataset

# Read the CSV file
file_path = r".\data\find.csv"
df = pd.read_csv(file_path)

# Delete rows where both column 2 (index 1) and column 127 (index 126) are **simultaneously** 0 or NaN
df = df[~((df[df.columns[1]].isna() | (df[df.columns[1]] == 0)) &
          (df[df.columns[126]].isna() | (df[df.columns[126]] == 0)))]

# Modify the label column to use the prefix of Image_ID as the label name
df['label'] = df['Image_ID'].apply(lambda x: str(x).split('_')[0])

# Save the processed data to a new file
new_file_path = r".\data1\change.csv"
df.to_csv(new_file_path, index=False)

print(f"Processing complete. Data has been saved to {new_file_path}")
