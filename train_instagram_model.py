import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the training dataset
df = pd.read_csv('train.csv')

# Separate the features (X) and the target label (y)
X = df.drop(columns=['fake'])
y = df['fake']

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Save the trained model to a file so your web app can use it
with open('instagram_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
    
print("Instagram model trained and saved successfully!")