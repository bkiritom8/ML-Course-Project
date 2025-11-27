# ML-Course-Project
Explainable Machine Learning for Alzheimer's Disease Stage Classification: Integrating Gene Expression, Clinical Features, and Early Onset Status

# Alzheimer's Disease Classification - Data Preprocessing

## Project Overview
Explainable Machine Learning for Alzheimer's Disease Stage Classification using gene expression data from blood samples.

## Current Status: Data Preprocessing Complete

## Dataset Information

### Source
- **GSE110226**: 20 patients (brain tissue, Affymetrix platform)
- **GSE63060**: 329 patients (blood samples, Illumina platform)
- **GSE85426**: Excluded (insufficient probe mapping)
- **ADNI**:     744 patients

### Final Dataset
- **Patients:** 1042 (152 AD, 371 Control, 519 MCI)
- **Features:** 1,003
  - 1,000 gene expression features
  - 1 clinical feature (Age)
  - 1 demographic feature (Sex)
- **Target:** Multi-class classification (AD, Control, MCI)
- **Class Balance:** Applied SMOTE.
     Final shape (1557, 1003)

## Preprocessing Pipeline

### 1. Gene Expression Preprocessing
- **Log2 transformation** - Handles skewed expression distributions
- **Quantile normalization** - Removes batch effects, harmonizes multi-platform data
- **Low-variance filtering** - Removed bottom 20% (16,196 → 12,938 genes)

### 2. Feature Selection
- **XGBoost-based importance ranking**
- **Top 1,000 genes selected** (12,938 → 1,000)

### 3. Clinical Features
- **Age:** Z-score standardization (mean=0, std=1)
- **Sex:** Binary encoding (Male=1, Female=0)
- **APOE:** Excluded (94% missing data)

### 4. Class Imbalance Handling
- **SMOTE applied** (1042 → 1557 patients)

## Files

### Data Files
- **`Final_Balanced_Dataset_SMOTE.csv`** - Preprocessed  Dataset with shape (1557, 1003)

### Code
- **`Final_Data_Preprocessing.ipynb`** - Complete preprocessing pipeline with documentation

## Feature Description

| Feature Type | Count | Description | Range |
|-------------|-------|-------------|-------|
| Gene Expression | 1,000 | Quantile normalized gene expression values | [0, 1] |
| Age | 1 | Z-score standardized age | ~[-3, 3] |
| Sex | 1 | Binary encoded (Male=1, Female=0) | {0, 1} |

## Data Quality
- No missing values
- No duplicate samples
- Balanced classes
- Features normalized and ready for ML

## Next Steps (For Model Building Team)
1. Load `Final_Data_Preprocessing`
2. Split the dataset into train & test set.
3. Perform train-test split (recommended: 80-20)
4. Apply 5-fold cross-validation × 10 repeats on training set
5. Train models:
   - Logistic Regression (L2 regularization)
   - SVM (RBF kernel)
   - Random Forest
   - XGBoost
   - Hybrid CNN-DNN with Self-Attention
6. Evaluate on test set

## Requirements
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Preprocessing Details

### Why These Methods?

**Log2 Transformation:**
- Compresses skewed gene expression distributions
- Stabilizes variance across expression levels
- Standard in genomics (enables fold-change interpretation)

**Quantile Normalization:**
- Removes batch effects between different datasets
- Makes all genes have identical distributions
- Essential for multi-platform integration
- Standard practice in gene expression analysis

**Z-score for Age (not quantile):**
- Places age on comparable scale with genes
- Preserves natural age relationships
- Standard for continuous clinical variables

**Binary Encoding for Sex:**
- Simple numerical representation
- Already on [0,1] scale

**SMOTE (not class weighting):**
- Creates synthetic samples to balance classes
- Achieved perfect 50-50 balance
- Class weighting skipped as data is now balanced

## Contact
Moumita Baidya/ baidya.m@northeastern.edu
```

---

#### **data_description.txt**
```
PREPROCESSED ALZHEIMER'S DATASET
=================================

FILE: X_FINAL_preprocessed.csv
-------------------------------
Shape: 1557 rows × 1,003 columns

Rows: Individual patients
- 519 Alzheimer's Disease patients
- 519 Control patients (no Alzheimer's)
- 519 MCI Patients

Columns: Features (1,002 total)
- Columns 0-999: Gene expression values
  - Source: Blood/tissue microarray data
  - Preprocessing: Log2 → Quantile normalization
  - Range: [0, 1] (quantile normalized)
  - Selected: Top 1,000 most important genes by XGBoost

- Column 1000: Age_Zscore
  - Original range: 37-89 years
  - Preprocessing: Z-score standardization
  - Range: Approximately [-3, 3]
  - Interpretation: (Age - Mean) / Std

- Column 1001: Sex_Male
  - Binary encoding: 1 = Male, 0 = Female
  - Range: {0, 1}

PREPROCESSING SUMMARY
---------------------
✓ Data sources merged and harmonized (GSE110226 + GSE63060)
✓ Gene expression: Log2 transformed + Quantile normalized
✓ Feature selection: 14907 → 1,000 genes
✓ Clinical features: Standardized and encoded
✓ Class balance: SMOTE applied
✓ Quality: No missing values, no duplicates

DATA IS READY FOR MACHINE LEARNING
-----------------------------------
- Train-test split: Not yet performed (to be done by modeling team)
- Cross-validation: Ready for k-fold CV
- Scaling: Already applied (quantile norm + z-score)
- No additional preprocessing needed

```

---

#### **requirements.txt**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.8.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---
