**ML-Course-Project: Explainable Machine Learning for Alzheimer's Disease Stage Classification**

**Overview**
--------

This project develops an interpretable machine learning system for multi-class Alzheimer's Disease stage classification using blood-based gene expression data. The system integrates data from multiple public sources (GSE63060, GSE85426, ADNI), applies rigorous batch correction, performs systematic algorithm comparison, and provides explainability through SHAP analysis.

Key Achievement: 68.60% accuracy on independent test set with strong model confidence (82.19% on correct predictions).


**Dataset Information**
-------------------

Source Datasets:
- GSE63060: 329 samples (blood, Illumina platform)
- GSE85426: 180 samples (blood, Affymetrix platform)
- ADNI: 700 samples (multiple platforms)
- Total: 1,209 samples across multiple research institutions

Final Dataset:
- Total Samples: 1,209 (967 training, 242 testing)
- Features: 1,003
  - 1,000 gene expression features (log2 transformed, quantile normalized)
  - 1 clinical feature (Age, Z-score standardized)
  - 1 demographic feature (Sex, binary encoded)
- Classes: 3 (AD: 235, MCI: 519, Control: 455)
- Data Quality: No missing values, no duplicates, all features normalized


**Installation and Setup**
----------------------

STEP 1: Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning repository)

Verify Python installation:
  python --version
  pip --version

STEP 2: Create Virtual Environment (Recommended)

Creating a virtual environment isolates project dependencies from system packages.

On macOS/Linux:
  python -m venv venv
  source venv/bin/activate

On Windows:
  python -m venv venv
  venv\Scripts\activate

After activation, your terminal should show (venv) at the beginning of the prompt.

STEP 3: Install All Required Packages

Install all dependencies from requirements.txt:
  pip install -r requirements.txt

STEP 4: Verify Installation

Test that all packages are installed correctly:
  python -c "import pandas, numpy, sklearn, xgboost, shap, combat; print('All packages installed successfully')"

Expected output:
  All packages installed successfully


**Running the Project**
-------------------

IMPORTANT: Execute the notebooks in the following order. Do not skip steps or run out of order.

STEP 1: Run Preprocessing Notebook FIRST

The preprocessing notebook prepares raw data for modeling. This step is mandatory and must run first.

Start Jupyter:
  jupyter notebook

In the browser, open: Preprocessing.ipynb

After preprocessing completes, verify the output file exists:
  Final_Dataset_ForModeling.csv (1,209 rows x 1,003 columns)

STEP 2: Run Modelling Notebook SECOND

Only after preprocessing completes successfully, run the modelling notebook.

In the same Jupyter window, open: Modelling.ipynb

Output files generated:
- 01_shap_importance_by_class.png
- 02_shap_beeswarm.png
- 03_shap_dependence.png
- 04_importance_comparison.png
- 05_confusion_matrix.png
- 06_per_class_metrics.png
- 07_confidence_analysis.png
- confusion_matrices.png
- Batch_Correction_Comparison.png
- results_summary.csv

**Viewing Results**
---------------

STEP 1: Review Jupyter Outputs

After modelling notebook completes, all results are displayed in the notebook:
- Performance metrics tables
- Algorithm comparison results
- SHAP analysis summaries
- Visualization figures

STEP 2: View Generated Figures

All visualization PNG files are saved in the project directory.

STEP 3: Read Final Report

Open the comprehensive project report:
  Project_Report.pdf


The report contains:
- Problem statement and motivation
- Related work and state of the art
- Detailed methodology
- Comprehensive results with tables and figures
- Discussion of findings
- Team contributions


**Project Results**
---------------

Algorithm Performance on Test Set (n=242):

Algorithm               Accuracy    Macro F1    Kappa      Training Time
Logistic Regression     57.9%       0.589       0.357      1.2 seconds
SVM (RBF)              63.6%       0.599       0.437      45.3 seconds
Random Forest          62.8%       0.604       0.425      23.1 seconds
XGBoost (Best)         68.6%       0.685       0.512      5.2 seconds
MLP Neural Network     57.9%       0.577       0.350      120.5 seconds

Per-Class Performance (XGBoost):

Class           Precision   Recall      F1-Score    Support
AD              0.655       0.809       0.724       46
Control         0.671       0.538       0.598       85
MCI             0.712       0.760       0.735       104

Overall Metrics:
- Accuracy: 68.60%
- Macro F1-Score: 0.6854
- Cohen's Kappa: 0.5115
- Model Confidence (correct predictions): 82.19%
- Model Confidence (incorrect predictions): 71.89%

Top Features (SHAP Importance):
1. OAZ1 (Polyamine synthesis) - 1.203
2. UBC (Protein quality control) - 0.524
3. RPF2 (Translational efficiency) - 0.368
4. Sex (Demographics) - 0.345
5. SMCHD1 (Chromatin regulation) - 0.313



**Project Structure**
-----------------

ML-Course-Project/
│
├── README.md                              (Project documentation)
├── requirements.txt                       (Python dependencies)
│
├── ML_Proposal.pdf                        (Original proposal)
├── Project_Report.pdf                     (Final report)
│
├── Preprocessing.ipynb                    (Data preprocessing - RUN FIRST)
├── Modelling.ipynb                        (Model training - RUN SECOND)
│
├── data/                                  (Raw datasets directory)
│   ├── ADNI/
│   │   ├── ADNI_combined_dataset.csv.gz
│   │   ├── ADNI_GEO_formatted.csv.gz
│   │   └── ADNI_gene_annotations.csv
│   │
│   └── GSE/
│       ├── GSE63060_series_matrix.txt.gz
│       ├── GSE85426_series_matrix.txt.gz
│       ├── GSE85426_normalized_data.txt
│       ├── broad_agilent_annotation.chip
│       └── GPL6947_annotation.csv
│
└── output/                                (Generated results directory)
    ├── Final_Dataset_ForModeling.csv
    ├── Final_Dataset_BatchCorrected.csv
    ├── Selected_Genes.csv
    ├── results_summary.csv
    │
    ├── Batch_Correction_Comparison.png
    ├── 01_shap_importance_by_class.png
    ├── 02_shap_beeswarm.png
    ├── 03_shap_dependence.png
    ├── 04_importance_comparison.png
    ├── 05_confusion_matrix.png
    ├── 06_per_class_metrics.png
    ├── 07_confidence_analysis.png
    │
    ├── batch_effects_before.png
    ├── batch_effects_after.png
    └── confusion_matrices.png


**Execution Workflow Summary**
--------------------------

The complete workflow is as follows:

1. Install Python 3.8+
2. Create virtual environment
3. Install requirements: pip install -r requirements.txt
4. Verify installation
5. Start Jupyter: jupyter notebook
6. FIRST: Run Preprocessing.ipynb completely
   - Wait for "Final_Dataset_ForModeling.csv" to be created
7. SECOND: Run Modelling.ipynb completely
   - Wait for all visualizations and results to be generated
8. Review results in notebook outputs
9. Open and review Project_Report.pdf

Do not skip steps or run notebooks out of order. The modelling notebook requires the output of the preprocessing notebook.


**Key Findings**
------------

Batch Correction Discovery:
Without batch correction, the model achieved only 55% macro F1-score (random-level performance). This was because the model was learning dataset origin rather than disease status. After applying ComBat batch correction, performance improved to 68.6% macro F1, a 24.6% absolute improvement. This critical finding highlights the importance of proper batch effect handling in multi-dataset studies.

Gene Count Optimization:
Initial proposal expected 1,000 genes to be optimal. Ablation study revealed 500 genes achieved better generalization (68.6% vs 67.1%) and faster training (5.2s vs 8.1s). This demonstrates the bias-variance trade-off: more features lead to overfitting on small test sets (n=242).

Algorithm Selection:
XGBoost significantly outperformed deep learning (MLP: 57.9% vs XGBoost: 68.6%). This validates recent literature (Grinsztajn et al. 2022) showing tree-based models outperform deep neural networks on tabular high-dimensional low-sample-size (HDLSS) data.

Blood-Based Biomarkers:
Classical Alzheimer's biomarkers (APOE, PSEN1) were largely absent or had very low expression in blood samples (APOE 94% missing). Our identified blood-accessible signatures (OAZ1, UBC, RPF2) represent systemic immune responses rather than brain-specific pathology, offering a complementary approach to traditional CSF and PET biomarkers.

Model Confidence Calibration:
The model shows appropriate uncertainty quantification with 82.19% confidence on correct predictions versus 71.89% on incorrect predictions. This 10.3% confidence gap supports clinical deployment where low-confidence predictions can trigger additional specialist review.


**Team Contributions**
------------------

Bhargav Pamidighantam - Explainability and Biological Analysis
- Comprehensive literature review (classical ML, deep learning, explainability)
- SHAP analysis (summary plots, beeswarm plots, dependence plots)
- Biological interpretation of identified biomarkers
- Discussion of findings and clinical implications

Akshatt Kain - Model Building and Optimization
- Algorithm comparison framework (5 algorithms with consistent evaluation)
- Bayesian hyperparameter optimization with repeated k-fold cross-validation
- Gene count ablation study (100-5000 genes, identified 500-gene optimum)
- Testing of MLP neural network (empirical evidence for tree-model superiority)
- Algorithm selection and hyperparameter configuration

Moumita Baidya - Data Preprocessing and Integration
- ComBat batch correction implementation (identified and solved 55% to 68.6% problem)
- Log2 transformation, quantile normalization, variance filtering
- SMOTE class balancing (applied to training data only, preventing data leakage)
- Multi-dataset integration (GSE63060, GSE85426, ADNI)
- Data quality assurance and experimental design


**References**
----------

1. WHO (2023). Dementia: Key Facts

2. Sarma M, Chatterjee S (2025). Machine Learning multiclassification for stage diagnosis of AD. Discoveries Applied Sciences, 7:636

3. Ali F, et al. (2021). Smart healthcare monitoring with ensemble deep learning. Information Fusion, 63:208-222

4. Wen J, et al. (2020). CNNs for classification of AD: Overview. Medical Image Analysis, 63:101694

5. Grinsztajn L, et al. (2022). Why tree-based models outperform DL on tabular data. Neural Information Processing Systems (NeurIPS)

6. Tong T, et al. (2017). Grading biomarker for MCI to AD conversion. IEEE Transactions on Biomedical Engineering, 64(1):155-165

7. Lunnon K, et al. (2013). Blood Gene Expression Marker of Early AD. Journal of Alzheimer's Disease, 33(3):669-677

8. Sperling RA, et al. (2014). The A4 Study: Stopping AD Before Symptoms Begin. Science Translational Medicine, 6(228):228fs13

9. Deo AS, et al. (2019). Blood-based transcriptomic biomarkers for early detection of mild cognitive impairment. Frontiers in Neuroscience, 13:199

10. Park S, et al. (2020). Gene expression profiles of blood-derived peripheral monocytes in mild cognitive impairment to Alzheimer's disease conversion. Scientific Reports, 10:3506


**Contact Information**
-------------------

Project Team:
- Bhargav Pamidighantam (pamidighantam.b@northeastern.edu)
- Akshatt Kain (kain.a@northeastern.edu)
- Moumita Baidya (baidya.m@northeastern.edu)

Institution:
Northeastern University, CS 6140 Machine Learning

Last Updated: December 2025
