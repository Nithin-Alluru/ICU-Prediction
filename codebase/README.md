
---

## Running the Project

### Prerequisites
Ensure you have Python 3.8 or later installed. Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

### Data Preparation
1. Place your dataset in the same directory as the project code and name it `dataset.csv` (or update the `pd.read_csv()` statement in the code accordingly).

---

### Execution Order
- **Phase 1: Feature Engineering & Exploratory Data Analysis (EDA)** must be executed first to preprocess the dataset.
- Once **Phase 1** is complete, the preprocessed dataset will be used for subsequent phases.
- **Phases 2, 3, and 4** can be executed in any order, as they do not depend on each other.

---

### Phase 1: Feature Engineering & Exploratory Data Analysis (EDA)
1. Import the dataset and apply data cleaning and preprocessing steps:
   - Handle missing values (numerical: median, categorical: mode, binary: 0).
   - Remove duplicates.
   - Perform one-hot encoding for categorical features.
   - Normalize numerical features using **StandardScaler**.
   - Reduce data dimensionality using:
     - Principal Component Analysis (PCA).
     - Singular Value Decomposition (SVD).
     - Random Forest Feature Importance.
     - High-correlation filtering.

2. Save the preprocessed dataset for use in subsequent phases.

---

### Phase 2: Regression Analysis
1. Set the target variable for regression (`pre_icu_los_days`).
2. Apply dimensionality reduction on features using high-correlation filtering and Random Forest feature importance.
3. Split the dataset into training and test sets.
4. Train a **Multiple Linear Regression Model** using:
   - Ordinary Least Squares (OLS).
   - Backward stepwise regression for feature selection.
5. Evaluate model performance:
   - Metrics: R-squared, Adjusted R-squared, AIC, BIC, MSE.
   - Visualize actual vs. predicted values.

---

### Phase 3: Classification Analysis
1. Set the target variable for classification (`hospital_death`).
2. Train and evaluate multiple classifiers:
   - **Decision Tree**: Pre-pruned and post-pruned.
   - **Logistic Regression**: Best parameters found using grid search.
   - **K-Nearest Neighbors (KNN)**: Optimal K found using the elbow method.
   - **Support Vector Machines (SVM)**: Linear, polynomial, and RBF kernels.
   - **Naïve Bayes**.
   - **Random Forest**: Bagging, stacking, and boosting methods.
   - **Neural Network**: Multi-layer Perceptron (MLP).
3. Evaluate classifiers using:
   - Confusion Matrix, Precision, Recall, F1-score, Specificity, ROC-AUC, and Stratified K-Fold Cross-Validation.
4. Select the best classifier based on performance metrics.

---

### Phase 4: Clustering and Association Rule Mining
1. **Clustering**:
   - Apply **K-Means Clustering**:
     - Use silhouette analysis and within-cluster sum of squares to determine the optimal number of clusters.
     - Evaluate clustering performance using Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, Adjusted Rand Index, and Mutual Information Score.
   - Apply **DBSCAN Clustering**:
     - Identify clusters and noise points.
     - Evaluate clustering performance with Silhouette Score and Davies-Bouldin Index.

2. **Association Rule Mining**:
   - Use the **Apriori Algorithm** on binary features to identify frequent itemsets.
   - Generate and sort association rules by confidence and lift.

---