# Predicting Exoplanet Candidates from Kepler Data

**Author:** Rahul singh bisht

**Live Project Link:** [**View the Google Colab Notebook Here**]([https://colab.research.google.com/drive/1JYGHHv0r42J7yUPHogBD7Rt9SG8sW-xq](https://colab.research.google.com/drive/1JYGHHv0r42J7yUPHogBD7Rt9SG8sW-xq?usp=sharing))

---

## 1. Project Objective

The goal of this project was to build a machine learning model to accurately classify planet candidates from the NASA Kepler space telescope.

The dataset contains "Kepler Objects of Interest" (KOI) which are labeled as "CONFIRMED" (a verified exoplanet) or "FALSE POSITIVE" (an object that mimics a planet). To create a clean binary classification problem, this project filters the data to only these two classes, building a model to predict a classification of **Class 1 (CONFIRMED)** or **Class 0 (FALSE POSITIVE)**.

## 2. Data Cleaning and Preparation

The initial dataset contained 7,316 rows and 44 columns. A significant challenge was the 1,100 rows (15.5% of the data) with missing values.

-   **Filtering:** All "CANDIDATE" rows were removed to create a binary problem.
-   **Column Removal:** Non-scientific identifier columns (e.g., `kepid`, `kepoi_name`) and columns with 100% null values (`koi_teq_err1`, `koi_teq_err2`) were dropped.
-   **Imputation:** Instead of dropping 15.5% of the dataset, missing `NaN` values were imputed using the **median** of each column. The median was chosen over the mean as it is more robust to the outliers common in astronomical data.

## 3. Data Leakage Investigation

A primary focus of this project was the identification and resolution of data leakage, a common problem that leads to unrealistic model performance.

### Attempt 1: The "0.99 Score" Trap (Feature Leakage)

The first model achieved an unrealistic 0.99 accuracy. This was a classic sign of data leakage.

-   **Cause:** The model was trained on features like `koi_score` and `koi_fpflag_...` (False Positive Flags). These columns are not raw measurements but are the *results* of a separate analysis, effectively giving the model the answers.
-   **Solution:** All pre-computed and non-raw measurement columns were dropped.

### Attempt 2: The Preprocessing Leak

A second, more subtle form of leakage was identified: imputing and scaling the *entire dataset* **before** splitting it into training and test sets. This causes information from the test set (e.g., its median) to "leak" into the training process, invalidating the test results.

## 4. The Corrected Workflow

To build a valid model, the pipeline was re-implemented in the correct order:

1.  **Split First:** The data was split into `x_train` and `x_test` **before** any other processing. The split correctly used `random_state=42` (for reproducibility) and `stratify=y` (to maintain the original class balance in both sets).
2.  **Impute Separately:** The median was calculated from the `x_train` set *only*. This median was then used to fill `NaN` values in *both* `x_train` and `x_test`.
3.  **Scale Separately:** A `StandardScaler` was fit *only* on the `cleaned_x_train` data. This fitted scaler was then used to `transform` both the training and test sets.

## 5. Final Models and Results

Three models were trained and evaluated on the properly processed data.

| Model | Training Accuracy | Test Accuracy |
| :--- | :--- | :--- |
| Logistic Regression | 0.9174 | 0.9071 |
| **SVM (Linear Kernel)** | **0.9215** | **0.9118** |
| SVM (RBF Kernel) | 0.8880 | 0.9098 |


## 6. Conclusion

The data is highly linearly separable, as shown by the strong performance of both the Logistic Regression and Linear SVM models.

The **Support Vector Machine (SVM) with a Linear Kernel** provided the best and most reliable performance, achieving a final **Test Accuracy of 91.18%**.

## 7. Technology Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn (LogisticRegression, SVC, StandardScaler, train_test_split)
