# model_utils.py
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


def load_dataset(path):
    df = pd.read_csv(path)
    return df


def build_preprocessor(df, target_col='Career'):
    # Determine feature columns
    X = df.drop(columns=[target_col])

    # Heuristic: numeric vs categorical
    numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_feats = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Imputers and encoders
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ])

    return preprocessor, numeric_feats, categorical_feats


def train_and_save(df, model_path='models/career_model.joblib', preproc_path='models/preprocessor.joblib', target_col='Career'):
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[target_col])

    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]

    preprocessor, numeric_feats, categorical_feats = build_preprocessor(df_clean, target_col=target_col)

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf.fit(X_train, y_train)

    # Evaluate
    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)
    print('Classification report:\n', report)

    # Save
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

    # Save preprocessor for potential direct use
    joblib.dump({'numeric': numeric_feats, 'categorical': categorical_feats}, preproc_path)
    print(f"Saved preprocessor meta to {preproc_path}")

    return clf


def load_model(model_path='models/career_model.joblib'):
    return joblib.load(model_path)