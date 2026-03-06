import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('datasets/Career_Path_Dataset_with_Career.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nCareer paths distribution:")
print(df['Suggested_Career_Path'].value_counts())

# Handle missing values
print("\nHandling missing values...")
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna('Unknown', inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Feature engineering - create new features
print("\nEngineering features...")

# Count number of technical skills
df['Technical_Skills_Count'] = df['Technical_Skills'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown' else 0
)

# Count soft skills
df['Soft_Skills_Count'] = df['Soft_Skills'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown' else 0
)

# Count languages
df['Languages_Count'] = df['Languages_Known'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) and x != 'Unknown' else 0
)

# Has certifications
df['Has_Certifications'] = df['Certifications'].apply(
    lambda x: 0 if pd.isna(x) or x == 'Unknown' or x == '' else 1
)

# Has work experience
df['Has_Experience'] = df['Past_Jobs_Internships'].apply(
    lambda x: 0 if pd.isna(x) or x == 'Unknown' or x == '' else 1
)

# Select features for model
categorical_features = [
    'Gender', 'Location', 'Highest_qualification', 'Stream',
    'Current_Academic_Level', 'Fields_of_Interest', 
    'Preferred_Work_Style', 'Work_Type_Interest', 'Willing_to_Relocate'
]

numerical_features = [
    'Age', 'Grade_CGPA_Percentage', 'Technical_Skills_Count',
    'Soft_Skills_Count', 'Languages_Count', 'Has_Certifications',
    'Has_Experience'
]

# Label encoding for categorical variables
label_encoders = {}
print("\nEncoding categorical variables...")

for feature in categorical_features:
    le = LabelEncoder()
    df[feature + '_encoded'] = le.fit_transform(df[feature].astype(str))
    label_encoders[feature] = le

# Prepare features and target
encoded_features = [f + '_encoded' for f in categorical_features]
all_features = numerical_features + encoded_features

X = df[all_features]
y = df['Suggested_Career_Path']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
label_encoders['target'] = le_target

print(f"\nFeatures used: {all_features}")
print(f"Number of features: {len(all_features)}")
print(f"\nTarget classes: {le_target.classes_}")

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Make predictions
print("\nEvaluating model...")
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test, 
                          target_names=le_target.classes_))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the model and encoders
print("\nSaving model and encoders...")
with open('career_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

with open('feature_names.pkl', 'wb') as f:
    pickle.dump(all_features, f)

print("\nModel training complete!")
print("Files saved:")
print("- career_model.pkl")
print("- label_encoders.pkl")
print("- feature_names.pkl")