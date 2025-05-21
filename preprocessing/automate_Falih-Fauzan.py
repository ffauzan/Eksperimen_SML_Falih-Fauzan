import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('./../crop_data_raw.csv')

# Preprocess
y = df['label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create a new DF with the encoded labels
df_cleaned = df.copy()
df_cleaned['label_encoded'] = y_encoded

# Save the cleaned DF to a new CSV file
df_cleaned.to_csv('crop_data_cleaned.csv', index=False)

