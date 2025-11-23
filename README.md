# Credit Card Default Prediction with CI/CD

Simple ML project with proper train/val/test split and CI/CD.

## Dataset

Credit Card Default from UCI Repository
- 30,000 samples
- 23 features
- Binary classification

**Download**: See download options in project guide

## Quick Start

```bash
# 1. Download dataset (use one of the options above)
wget https://raw.githubusercontent.com/YuChenAmberLu/Data-Science--Credit-Card-Default/master/UCI_Credit_Card.csv
mv UCI_Credit_Card.csv data/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train
python train.py

# 4. Test
python test.py
```

## Results

Expected performance:
- Accuracy: ~82%
- AUC: ~77%
- F1-Score: ~45%

## CI/CD

Push to GitHub and watch Actions run automatically!


## 🚀 Automated CI/CD Pipeline
