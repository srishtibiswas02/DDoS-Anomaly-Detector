# Import necessary libraries
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import numpy as np

# Step 1: Load and Concatenate CSV Files from the CICIDS 2017 Folder
def load_data(folder_path):
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder path '{folder_path}' does not exist. Please check the path and try again.")
    
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not all_files:
        raise ValueError(f"No CSV files found in the folder '{folder_path}'. Please check the folder content.")
    
    data_frames = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df.rename(columns=lambda x: x.strip(),inplace=True)
            # Apply strip function to all column names to remove any leading or trailing whitespace
            df.columns = [col.strip() for col in df.columns]
            
            # Check if 'Label' column exists in the current CSV file
            if 'Label' in df.columns:
                data_frames.append(df)
            else:
                print(f"Skipping {file}: 'Label' column not found.")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not data_frames:
        raise ValueError("No valid CSV files with the 'Label' column were found. Please check the dataset.")
    
    # Concatenate all data frames
    combined_data = pd.concat(data_frames, ignore_index=True)
    return combined_data

# Specify the folder path containing the CICIDS 2017 CSV files
folder_path = 'datafolder'  # Replace with your folder path
data = load_data(folder_path)    

# Step 2: Preprocess the Data
# Drop unnecessary columns and handle missing values
data = data.drop(columns=['Flow ID', 'Timestamp'], errors='ignore')
data = data.fillna(0)

# Ensure 'Label' column exists in the concatenated data
if 'Label' not in data.columns:
    raise ValueError("The 'Label' column is missing from the concatenated dataset. Please check the CSV files.")

# Encode the target variable (Label) as 0 for BENIGN and 1 for attacks
data['Label'] = data['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# Select relevant features for training
# selected_features = [
#     'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 
#     'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Bwd Packet Length Max', 'Flow Bytes/s', 
#     'Flow Packets/s', 'Fwd IAT Total', 'Bwd IAT Total', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward'
# ]

# Replace infinite values and values too large for float64
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)

# Split dataset into features (X) and labels (y)
X = data.drop(columns=['Label'])
y = data['Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=10)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Step 3: Save the Model and Scaler
dump(model, 'random_forest_model.joblib')
dump(scaler, 'scaler.joblib')
