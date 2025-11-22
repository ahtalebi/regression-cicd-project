
"""
Train Logistic Regression Model on Credit Card Default Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
import os

def load_data():
    """Load the credit card default dataset"""
    print("📊 Loading Credit Card Default dataset...")
    
    # Read the CSV file
    df = pd.read_csv('data/UCI_Credit_Card.csv')
    
    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n🔧 Preprocessing data...")
    
    # Drop ID column if exists
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Separate features and target
    # The target is usually the last column (default.payment.next.month or similar)
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def split_data(X, y):
    """Split data into train, validation, and test sets"""
    print("\n✂️ Splitting data...")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: 75% train, 25% val (of the 80%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(X_train, y_train):
    """Train logistic regression model"""
    print("\n🚀 Training Logistic Regression model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("✅ Model training complete!")
    
    return model, scaler

def evaluate_model(model, scaler, X, y, dataset_name=""):
    """Evaluate model performance"""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred)),
        'recall': float(recall_score(y, y_pred)),
        'f1_score': float(f1_score(y, y_pred)),
        'roc_auc': float(roc_auc_score(y, y_proba))
    }
    
    print(f"\n📊 {dataset_name} Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics

def save_model_and_metrics(model, scaler, train_metrics, val_metrics, test_metrics):
    """Save model, scaler, and metrics"""
    print("\n💾 Saving model and metrics...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(model, 'models/logistic_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("  ✅ Model saved to models/logistic_model.pkl")
    print("  ✅ Scaler saved to models/scaler.pkl")
    print("  ✅ Metrics saved to models/metrics.json")

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("🚀 CREDIT CARD DEFAULT PREDICTION - TRAINING PIPELINE")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Preprocess
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train model
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(model, scaler, X_train, y_train, "TRAIN")
    val_metrics = evaluate_model(model, scaler, X_val, y_val, "VALIDATION")
    test_metrics = evaluate_model(model, scaler, X_test, y_test, "TEST")
    
    # Save everything
    save_model_and_metrics(model, scaler, train_metrics, val_metrics, test_metrics)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
