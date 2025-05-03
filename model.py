import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv("Indian_earthquake.csv")

# Ensure dataset has required columns
required_columns = {'Magnitude', 'Depth', 'Latitude', 'Longitude'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"❌ Dataset is missing required columns: {required_columns - set(df.columns)}")

# ✅ Automatically Assign Risk Levels Based on Magnitude
def classify_risk(magnitude):
    if magnitude <= 3.0:
        return 'Low'
    elif 3.1 <= magnitude <= 5.0:
        return 'Medium'
    else:
        return 'High'

df['Risk_Level'] = df['Magnitude'].apply(classify_risk)

# Prepare features and labels
X = df[['Magnitude', 'Depth', 'Latitude', 'Longitude']]
y = df['Risk_Level']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model correctly
with open("earthquake_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully!")
