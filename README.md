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

## Files

### Data Files
- **`Final__Dataset.csv`** - Preprocessed  Dataset with shape (1557, 1003)

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
- Features normalized and ready for ML

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

## Contact
Moumita Baidya/ baidya.m@northeastern.edu
```

