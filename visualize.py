"""
Visualize model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_model_and_data():
    """Load trained model and data"""
    print("📊 Loading model and data...")
    
    # Load model and scaler
    model = joblib.load('models/logistic_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    # Load data
    df = pd.read_csv('data/UCI_Credit_Card.csv')
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    target_col = df.columns[-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Load metrics
    with open('models/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    print("✅ Loaded successfully!")
    return model, scaler, X, y, metrics

def plot_confusion_matrix(y_true, y_pred, ax, title):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

def plot_roc_curve(y_true, y_proba, ax, title):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

def plot_feature_importance(model, feature_names, ax):
    """Plot feature importance"""
    # Get coefficients
    importance = np.abs(model.coef_[0])
    indices = np.argsort(importance)[::-1][:15]  # Top 15
    
    ax.barh(range(len(indices)), importance[indices], color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Absolute Coefficient Value')
    ax.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

def plot_prediction_distribution(y_true, y_proba, ax):
    """Plot prediction probability distribution"""
    # Separate probabilities by actual class
    default_probs = y_proba[y_true == 1]
    no_default_probs = y_proba[y_true == 0]
    
    ax.hist(no_default_probs, bins=50, alpha=0.6, label='No Default (0)', color='green')
    ax.hist(default_probs, bins=50, alpha=0.6, label='Default (1)', color='red')
    ax.set_xlabel('Predicted Probability of Default')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_metrics_comparison(metrics, ax):
    """Plot metrics comparison across train/val/test"""
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    sets = ['train', 'validation', 'test']
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    for i, dataset in enumerate(sets):
        values = [metrics[dataset][m] for m in metric_names]
        ax.bar(x + i*width, values, width, label=dataset.capitalize())
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Across Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

def create_visualizations():
    """Create all visualizations"""
    print("\n🎨 Creating visualizations...")
    
    # Load everything
    model, scaler, X, y, metrics = load_model_and_data()
    
    # Make predictions
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    plot_confusion_matrix(y, y_pred, ax1, 'Confusion Matrix')
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    plot_roc_curve(y, y_proba, ax2, 'ROC Curve')
    
    # 3. Feature Importance
    ax3 = plt.subplot(2, 3, 3)
    plot_feature_importance(model, X.columns.tolist(), ax3)
    
    # 4. Prediction Distribution
    ax4 = plt.subplot(2, 3, 4)
    plot_prediction_distribution(y, y_proba, ax4)
    
    # 5. Metrics Comparison
    ax5 = plt.subplot(2, 3, 5)
    plot_metrics_comparison(metrics, ax5)
    
    # 6. Model Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    MODEL SUMMARY
    
    Dataset: Credit Card Default
    Samples: {len(X):,}
    Features: {X.shape[1]}
    
    TEST SET PERFORMANCE:
    • Accuracy:  {metrics['test']['accuracy']:.3f}
    • Precision: {metrics['test']['precision']:.3f}
    • Recall:    {metrics['test']['recall']:.3f}
    • F1-Score:  {metrics['test']['f1_score']:.3f}
    • ROC-AUC:   {metrics['test']['roc_auc']:.3f}
    
    Class Distribution:
    • No Default (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)
    • Default (1):    {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)
    
    Model: Logistic Regression
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=12, 
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/model_performance.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to plots/model_performance.png")
    
    # Create individual plots for README
    create_simple_plots(y, y_pred, y_proba, metrics)

def create_simple_plots(y, y_pred, y_proba, metrics):
    """Create simple individual plots"""
    
    # ROC Curve only
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_curve(y, y_proba, ax, 'ROC Curve - Credit Card Default Prediction')
    plt.savefig('plots/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix only
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(y, y_pred, ax, 'Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Saved individual plots to plots/")

def main():
    """Main execution"""
    print("="*60)
    print("📊 MODEL PERFORMANCE VISUALIZATION")
    print("="*60)
    
    create_visualizations()
    
    print("\n" + "="*60)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  📊 plots/model_performance.png")
    print("  📈 plots/roc_curve.png")
    print("  📉 plots/confusion_matrix.png")

if __name__ == "__main__":
    main()
