# Titanic Dataset: Data Cleaning and Preprocessing Pipeline

This repository contains a structured End-to-End Data Cleaning and Preprocessing pipeline for the classic **Titanic: Machine Learning from Disaster** dataset. The primary focus of this project is to transform raw, noisy demographic and travel data into a highly optimized, fully numerical dataset tailored for predictive machine learning models.

## Project Overview
Raw datasets are frequently plagued by missing records, irrelevant high-cardinality features, and text strings that machines cannot interpret. This project systematically resolves these challenges on the Titanic passenger log, increasing data quality through algorithmic imputation, surgical feature reduction, custom feature extraction, and categorical encoding.

## Pipeline Phases

### Part 1: Data Cleaning & Missing Value Imputation
The script evaluates missing value thresholds across 891 entries and applies distinct statistical recovery methods:
- **`Cabin` Column Drop:** Removed entirely since roughly **77.1%** of its data was missing, making it statistically unviable.
- **`Age` Imputation:** Patched missing values (**19.9%**) using the data subset's **median age (28.0)** to protect against outliers.
- **`Embarked` Imputation:** Replaced missing rows (**0.2%**) with the dataset's **mode ("S")**.
- **`Fare` Guard:** Automated structural safeguard checking to swap any hypothetical empty fares with the median price tag.

### Part 2: Feature Engineering & Preprocessing
To maximize predictive capability, new contextual signals were generated from raw text data:
- **Title Extraction:** Extracted titles (e.g., *Mr, Mrs, Miss*) out of names. High-variance or noble variants (*Lady, Capt, Dr, Rev, etc.*) were collapsed into a unified `'Rare'` class, while smaller duplicates (*Mlle, Mme*) were normalized.
- **Family Metrics (`FamilySize` & `IsAlone`):** Blended `SibSp` (siblings/spouses) and `Parch` (parents/children) into a unified calculation to categorize whether a traveler was entirely alone or part of a family unit.
- **Dimensionality Reduction:** Pruned high-cardinality or redundant columns (`PassengerId`, `Name`, `Ticket`, `SibSp`, `Parch`).

### Part 3: Categorical One-Hot Encoding
To adapt the data for machine learning models, categorical text blocks were transformed into binary indicator matrices via dummy encoding (`pd.get_dummies`), using `drop_first=True` to eliminate multi-collinearity (the dummy variable trap). 

## Final Feature Structure
The finished preprocessing loop shrinks or transforms columns down into a compact layout containing 13 features:

| Feature Name | Data Type | Description |
| :--- | :--- | :--- |
| **Survived** | `int64` | Target label (0 = Deceased, 1 = Survived) |
| **Age** | `float64` | Imputed passenger age |
| **Fare** | `float64` | Ticket price paid |
| **FamilySize** | `int64` | Total count of family members onboard |
| **IsAlone** | `int64` | Binary flag indicating isolated travelers (1 = Alone, 0 = Group) |
| **Sex_male** | `bool` | True if Male, False if Female |
| **Embarked_Q / _S** | `bool` | One-hot encoded structural port origins (Queenstown, Southampton) |
| **Title_Mr / _Mrs / _Rare**| `bool` | Extracted and grouped social status titles |
| **Pclass_2 / _Pclass_3** | `bool` | Socioeconomic socio-strata indicators (Second/Third Class) |

## Tech Stack
- **Language:** Python
- **Libraries:** `pandas`
- **Development Environment:** Jupyter Notebooks
