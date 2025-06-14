# DIA-ML-Prediction

This repository contains all code, data, and supporting files used in the project titled:  
**"Predictive Modeling of Drug-Induced Autoimmunity: A Machine Learning and Descriptor-Based Approach"**  
as part of the ANLY 699 Capstone Course.

## Project Overview

Drug-induced autoimmunity (DIA) poses a serious challenge in pharmaceutical safety. This study presents a reproducible machine learning pipeline that uses molecular descriptors derived from RDKit to predict the risk of DIA. An XGBoost classifier was trained on publicly available data and optimized using Optuna, with class imbalance addressed via SMOTE.


## 📁 Repository Structure

| File/Folder                        | Description |
|-----------------------------------|-------------|
| `ML for DIA prediction.ipynb`     | Google Colab notebook containing model development and training |
| `Proposal_Final.Rmd`              | RMarkdown file with full analysis and APA-style writeup |
| `ANLY 699 ML approach for DIA.pdf`| Final compiled paper in PDF format |
| `xgboost_final_model.pkl`         | Trained XGBoost model |
| `scaler.pkl`                      | Scaler object used for feature normalization |
| `DIA_trainset_RDKit_descriptors.csv` | Training dataset from InterDIA |
| `DIA_testset_RDKit_descriptors.csv`  | Test dataset from InterDIA |
| `ROC_Curve.png` / other figures   | Model performance visualizations |
| `Bibliography.bib`                | BibTeX file with all citations used in the paper |
| `README.md`                       | You're here! Project summary and usage |


##  Getting Started

To reproduce the results:

1. Clone the repository or download the ZIP
2. Open `ML for DIA prediction.ipynb` in Google Colab or Jupyter
3. Install required packages:  
   `xgboost`, `optuna`, `scikit-learn`, `rdkit`, `pandas`, `matplotlib`, `seaborn`
4. Run all cells and modify parameters as needed
5. R users can compile the `.Rmd` using `papaja::apa6_pdf`

## 📊 Results Summary

- Accuracy: ~87.5%  
- AUC: ~0.63  
- Moderate recall for DIA-positive class  
- Top predictive features were derived from structural descriptors (e.g., TPSA, PEOE_VSA2)

## Code & Data Availability

All data used in this study was sourced from the **InterDIA** dataset published in Huang et al. (2025) via *Toxicology*.  
The trained model, preprocessing pipeline, and all code files are included in this repository.


## Future Work

This model can be extended through:
- Integration of biological or transcriptomic data
- SHAP-based interpretability
- Adaptation to food safety risk prediction


##  License

This repository is shared under the MIT License.


##  Contact

**Madhura Bokil**  
[LinkedIn](https://www.linkedin.com/in/madhura-bokil) | mb2663@cornell.edu  
Capstone project for ANLY 699 – Harrisburg University

