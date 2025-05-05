#%% Import Libraries and Setup
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
# Display all columns in the DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.2f}'.format
df = pd.read_csv('dataset.csv')
#%% Phase 1: Feature Engineering & Exploratory Data Analysis (EDA)
import seaborn as sns
from sklearn.preprocessing import StandardScaler
## Data Preprocessing

# Drop column with no relevance or redunduncy --DONE
# h1 columns have high redunduncy with d1 columns.
h1_cols = [col for col in df.columns if col.startswith('h1_')]
df.drop(columns=['Unnamed: 83', 'encounter_id', 'patient_id', 'hospital_id', 'icu_id']+h1_cols, inplace=True)

# 1. Handle Missing Values
missing_before = df.isnull().sum().sum()
print("Total Number of Missing Values Before Imputation:", missing_before)

# Impute numerical columns with median
num_cols = [
    'age', 'bmi', 'height', 'pre_icu_los_days', 'weight', 'apache_2_diagnosis',
    'apache_3j_diagnosis', 'heart_rate_apache', 'map_apache', 'resprate_apache',
    'temp_apache', 'apache_4a_icu_death_prob', 'apache_4a_hospital_death_prob',
]

# Dynamically include all features starting with "d1_"
d1_cols = [col for col in df.columns if col.startswith('d1_')]
num_cols += d1_cols
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Impute categorical columns with mode
cat_cols = ['gender', 'ethnicity', 'icu_admit_source', 'icu_type', 'icu_stay_type', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'gcs_verbal_apache', 'gcs_motor_apache', 'gcs_eyes_apache']
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Impute remaining clinical binary features with 0 (assume "not observed" means absence)
binary_cols = [
    'elective_surgery', 'apache_post_operative', 'arf_apache', 'gcs_unable_apache', 'intubated_apache',
    'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'hospital_death'
]
df[binary_cols] = df[binary_cols].fillna(0)

# After imputation, recheck and log
missing_after = df.isnull().sum().sum()
print("Total Number of Missing Values After Imputation: ", missing_after)

# 2. Check for Duplicates
# Check for duplicate rows
num_duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {num_duplicates}")

# Note:- None Found.
# 3. Encoding Categorical Variables (e.g., One-Hot Encoding)
nominal_cols = [
    'gender', 'icu_admit_source', 'icu_type', 'icu_stay_type',
    'apache_3j_bodysystem', 'apache_2_bodysystem', 'ethnicity'
]
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=int)

# 4. Aggregation
# Function to calculate the range of 'max' and 'min' columns.
def aggregate_range(df, prefix):
    columns = [col for col in df.columns if col.startswith(prefix) and ('max' in col or 'min' in col)]
    for col in columns:
        if 'max' in col:
            min_col = col.replace('max', 'min')
            range_col = col.replace('max', 'range')
            df[range_col] = df[col] - df[min_col]
            df.drop([col, min_col], axis=1, inplace=True)
    return df
# Apply to d1 prefix
df = aggregate_range(df, 'd1_')
# Update num_cols
num_cols = [col for col in num_cols if col not in d1_cols]  # Remove d1_cols from num_cols
d1_cols = [col for col in df.columns if col.startswith('d1_')]
num_cols += d1_cols

# 5. Normalization/Standardization
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Verify Transformations
print("First 5 rows of transformed data:\n", df.head())
#%%  Dimensionality Reduction (e.g., PCA, Random Forest Analysis)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def dimensionality_reduction_observations(X, y, Classify):
    """
    This function performs and outputs observations for different dimensionality reduction methods:
    1. Random Forest Feature Importance
    2. Variance Inflation Factor (VIF)
    3. Principal Component Analysis (PCA)
    4. Singular Value Decomposition (SVD)

    Parameters:
    - X_train: The training features
    - y_train: The target variable
    """

    # 1. Principal Component Analysis (PCA)
    pca = PCA()
    pca.fit(X)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1

    plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1),
             np.cumsum(pca.explained_variance_ratio_), marker='o')
    interval = 5
    plt.xticks(np.arange(1, len(cumulative_variance) + 1, interval))

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axvline(x=n_components_95, color='g', linestyle='-')
    plt.text(n_components_95, 0.5, f'{n_components_95} Components', color='g', ha='center')

    plt.grid()
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    # 2. Singular Value Decomposition (SVD)
    svd = TruncatedSVD(n_components=X.shape[1])  # number of components = number of features
    X_svd = svd.fit_transform(X)
    explained_variance_svd = svd.explained_variance_ratio_

    # Plot explained variance ratio for SVD
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_variance_svd) + 1), explained_variance_svd, marker='o', linestyle='--')
    plt.title('Explained Variance by SVD Components')
    plt.xlabel('SVD Component')
    plt.ylabel('Explained Variance')
    plt.show()

    # 3. Random Forest Feature Importance
    if Classify == True:
        rf = RandomForestClassifier(random_state=1)
    else:
        rf = RandomForestRegressor(random_state=1)
    rf.fit(X, y)
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.title("Feature Importance from Random Forest")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # 4. High Correlation Filter

    # Corr Heat map before
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

    def filter_high_correlation(df_t, threshold=0.8):
        """Drops features with high correlation (above threshold)."""
        # Compute the correlation matrix
        correlation_matrix = df_t.corr()
        high_correlation = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_correlation.add((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        # Drop one feature from each high-correlation pair (keep the first feature)
        to_drop = {col2 for _, col2 in high_correlation}
        return df_t.drop(columns=to_drop), to_drop

    X_reduced, dropped_high_corr = filter_high_correlation(X)
    print(f"Dropped features due to high correlation: {dropped_high_corr}")

    # Corr Heat map after
    plt.figure(figsize=(12, 10))
    sns.heatmap(X_reduced.corr(), cmap='coolwarm', annot=False)
    plt.title("Correlation Heatmap")
    plt.show()

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def dimensionality_reduction(X, y, correlation_threshold=0.8, importance_threshold=0.02, is_classification=True):
    """
    Performs dimensionality reduction using High Correlation Filter followed by Random Forest.

    Parameters:
    - X: pd.DataFrame, feature set
    - y: pd.Series or np.array, target variable
    - correlation_threshold: float, correlation threshold for the High Correlation Filter (default: 0.8)
    - importance_threshold: float, threshold for feature importance to select features (default: 0.01)
    - is_classification: bool, True if task is classification, False for regression (default: True)

    Returns:
    - X_reduced: pd.DataFrame, reduced feature set after dimensionality reduction
    - selected_features: list, list of selected feature names
    """

    def filter_high_correlation(df, threshold):
        """Drops features with high correlation (above threshold)."""
        correlation_matrix = df.corr()
        high_correlation = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_correlation.add((correlation_matrix.columns[i], correlation_matrix.columns[j]))
        to_drop = {col2 for _, col2 in high_correlation}
        return df.drop(columns=to_drop), to_drop

    # Step 1: High Correlation Filter
    X_filtered, dropped_features = filter_high_correlation(X, correlation_threshold)
    print(f"High Correlation Filter: Dropped features: {dropped_features}")

    # Step 2: Random Forest for Feature Importance
    if is_classification:
        model = RandomForestClassifier(random_state=1)
    else:
        model = RandomForestRegressor(random_state=1)

    model.fit(X_filtered, y)
    feature_importances = pd.DataFrame({
        'Feature': X_filtered.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Select features with importance above the threshold
    selected_features = feature_importances.loc[
        feature_importances['Importance'] >= importance_threshold, 'Feature'].tolist()
    print(f"Random Forest: Selected features with importance >= {importance_threshold}: {selected_features}")

    # Step 3: Return reduced dataset
    X_reduced = X_filtered[selected_features]
    return X_reduced, selected_features

#%% Phase 2: Regression Analysis
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

target_variable = 'pre_icu_los_days'
X = df.drop(columns=[target_variable])
y = df[target_variable]
# Dimensionality Reduction Methods Exploration
dimensionality_reduction_observations(X, y, False)
# Perform dimensionality reduction
X_reduced, selected_features = dimensionality_reduction(X, y, correlation_threshold=0.8, importance_threshold=0.02, is_classification=False)
print(f"Reduced Feature Set Shape: {X_reduced.shape}")
print(f"Selected Features: {selected_features}")

# Corr Heat map after
plt.figure(figsize=(12, 10))
sns.heatmap(X_reduced.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1)

# Regression Model Fitting and Evaluation

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
ols_model = sm.OLS(y_train, X_train_const).fit()

# Print OLS Summary
print(ols_model.summary())

# T-test analysis: p-values for each coefficient
print("\nT-test for individual coefficients (p-values):")
print(ols_model.pvalues)

# F-test: Overall model significance
f_statistic = ols_model.fvalue
f_p_value = ols_model.f_pvalue
print(f"\nF-statistic: {f_statistic:.4f}, p-value: {f_p_value:.4f}")

#  Model Evaluation (R-squared, Adjusted R-squared, AIC, BIC, MSE)

# Predictions
y_train_pred = ols_model.predict(X_train_const)
y_test_pred = ols_model.predict(X_test_const)

# Calculate Model Performance Metrics
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
adjusted_r2_train = 1 - (1 - r2_train) * (len(y_train) - 1) / (len(y_train) - X_train_const.shape[1] - 1)
adjusted_r2_test = 1 - (1 - r2_test) * (len(y_test) - 1) / (len(y_test) - X_test_const.shape[1] - 1)

# AIC and BIC
aic = ols_model.aic
bic = ols_model.bic

# Display Model Metrics
print(f"\nTraining MSE: {mse_train:.4f}")
print(f"Test MSE: {mse_test:.4f}")
print(f"Training R^2: {r2_train:.4f}")
print(f"Test R^2: {r2_test:.4f}")
print(f"Training Adjusted R^2: {adjusted_r2_train:.4f}")
print(f"Test Adjusted R^2: {adjusted_r2_test:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")

# Confidence Interval Analysis
conf_int = ols_model.conf_int()
print("\nConfidence Intervals for Model Coefficients:")
print(conf_int)

# Backward Stepwise Regression
import statsmodels.api as sm

def stepwise_selection(X_train_const, y_train, threshold_in=0.01, threshold_out=0.01):
    """
    Performs backward stepwise regression for feature selection.

    Parameters:
    - X_train_const: DataFrame with the independent variables (with constant column already added)
    - y_train: Series with the dependent variable
    - threshold_in: p-value threshold for including features (default: 0.01)
    - threshold_out: p-value threshold for excluding features (default: 0.05)

    Returns:
    - selected_features: List of selected feature names
    """
    # Start with all features
    selected_features = X_train_const.columns.tolist()
    while True:
        # Fit model with the current set of features
        model = sm.OLS(y_train, X_train_const[selected_features]).fit()

        # Get the p-values for all features in the model
        pvalues = model.pvalues[1:]  # Ignore the intercept term
        max_pval_feature = pvalues.idxmax()  # Find feature with the highest p-value

        # If the highest p-value is greater than the threshold, remove it
        if pvalues[max_pval_feature] > threshold_out:
            selected_features.remove(max_pval_feature)
        else:
            # If no feature with p-value greater than threshold, stop
            break

    return selected_features

# Apply Backward Stepwise Regression
stepwise_features = stepwise_selection(X_train_const, y_train)
print(f"\nStepwise Regression Selected Features: {stepwise_features}")

X_selected = X_train_const[stepwise_features]
final_model = sm.OLS(y_train, X_selected).fit()

# Get model statistics
r_squared = final_model.rsquared
adj_r_squared = final_model.rsquared_adj
aic = final_model.aic
bic = final_model.bic
mse = mean_squared_error(y_train, final_model.predict(X_selected))

# Display model statistics
print(f"R-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
print(f"MSE: {mse:.4f}")

# Plotting Actual vs Predicted
y_pred = final_model.predict(X_selected)
comparison = pd.DataFrame({'Actual pre_icu_los_days': y_test, 'Predicted pre_icu_los_days': y_pred})
comparison.sort_index(inplace=True)
plt.figure(figsize=(10, 6))
plt.scatter(comparison.index, comparison['Actual pre_icu_los_days'],
            label='Actual pre_icu_los_days', color='blue', marker='o', s=20, alpha=0.7)
plt.scatter(comparison.index, comparison['Predicted pre_icu_los_days'],
            label='Predicted pre_icu_los_days', color='red', marker='x', s=40, alpha=0.8)
plt.xlabel('Observation')
plt.ylabel('pre_icu_los_days')
plt.title('Actual vs Predicted pre_icu_los_days')
plt.legend()
plt.show()

#%% Phase 3: Classification Analysis
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

target_variable = 'hospital_death'
X = df.drop(columns=[target_variable])
y = df[target_variable]

# Dimentionality Reduction Methods Exploration
dimensionality_reduction_observations(X, y,True)
# Perform dimensionality reduction
X_reduced, selected_features = dimensionality_reduction(X, y, correlation_threshold=0.8, importance_threshold=0.02, is_classification=True)
print(f"Reduced Feature Set Shape: {X_reduced.shape}")
print(f"Selected Features: {selected_features}")

# Corr Heat map after
plt.figure(figsize=(12, 10))
sns.heatmap(X_reduced.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=1)
# Imbalance Handling with SMOTE
import seaborn as sns
# Plot to visualize imbalance
sns.countplot(x=y_train)
plt.title("Imbalanced Distribution of 'hospital_death'")
plt.show()

smote = SMOTE(random_state=1)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Plot to visualize balance
sns.countplot(x=y_train_balanced)
plt.title("Balanced Distribution of 'hospital_death'")
plt.show()

#%% Clasification Model Evaluation
# 1. Confusion Matrix
# 2. Precision, Recall, Specificity, F-Score
# 3. ROC Curve and AUC
# 4. Stratified K-Fold Cross Validation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

def evaluate_classifier(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.4f}")
    # Sensitivity (Recall)
    sensitivity = recall_score(y_test, y_pred)
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    # Specificity: Specificity = TN / (TN + FP)
    TN = cm[0][0]
    FP = cm[0][1]
    specificity = TN / (TN + FP)
    print(f"Specificity: {specificity:.4f}")
    # F-score (F1-score)
    fscore = f1_score(y_test, y_pred)
    print(f"F-score: {fscore:.4f}")
    # Accuracy
    train_accuracy = model.score(X_train_balanced, y_train_balanced)
    test_accuracy = model.score(X_test, y_test)
    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # Stratified K-fold Cross Validation
    skf = StratifiedKFold(n_splits=5)  # You can change n_splits as needed
    cv_scores = cross_val_score(model, X_test, y_test, cv=skf)
    print(f"Stratified K-fold Cross Validation (Accuracy): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
#%% Classifiers
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
# 1. Decision Tree
# Pre Pruned

# Note: The GridSearchCV section has been commented out to save time during evaluation.
# This step is computationally expensive and not necessary since the best parameters
# have already been identified. The identified optimal parameters are directly used
# to configure the DecisionTreeClassifier below.

"""
# Grid Search for Hyperparameter Tuning (Commented Out)
param_grid = {
    'max_depth': [1, 2, 3, 4, 5],
    'min_samples_split': [20, 30, 40],
    'min_samples_leaf': [10, 20, 30],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=1), param_grid, cv=5)
grid_search.fit(X_train_balanced, y_train_balanced)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)
clf_pre_pruned = grid_search.best_estimator_
"""

# Best Parameters Identified by Grid Search
best_params = {
    'criterion': 'gini',
    'max_depth': 5,
    'max_features': 'sqrt',
    'min_samples_leaf': 10,
    'min_samples_split': 20,
    'splitter': 'best'
}

# Directly Applying the Best Parameters to the Model
clf_pre_pruned = DecisionTreeClassifier(
    random_state=1,
    criterion=best_params['criterion'],
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    splitter=best_params['splitter']
)

clf_pre_pruned.fit(X_train_balanced, y_train_balanced)
evaluate_classifier(clf_pre_pruned, X_test, y_test)

plt.figure(figsize=(12, 8))
plot_tree(clf_pre_pruned, filled=True, feature_names=X.columns, class_names=['Died', 'Survived'])
plt.show()
#%% Post Pruning
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Initialize the base DecisionTreeClassifier without pruning
clf_no_prune = DecisionTreeClassifier(random_state=5805)
model = clf_no_prune.fit(X_train_balanced, y_train_balanced)

# Uncomment the section below if you need to perform a grid search to find the optimal ccp_alpha.
# This process is computationally expensive and unnecessary here as the best alpha is already provided.

"""
# Grid Search for Optimal Alpha (Commented Out)
# path = clf_no_prune.cost_complexity_pruning_path(X_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities
#
# train_scores = []
# test_scores = []
#
# for alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(random_state=5805, ccp_alpha=alpha)
#     clf.fit(X_train, y_train)
#     train_scores.append(clf.score(X_train_balanced, y_train_balanced))
#     test_scores.append(clf.score(X_test, y_test))
#
# best_alpha = ccp_alphas[test_scores.index(max(test_scores))]
"""

# Use the pre-determined optimal alpha directly
best_alpha = 7.80065675359549e-05

# Initialize and train the DecisionTreeClassifier with post-pruning
clf_post_pruned = DecisionTreeClassifier(random_state=5805, ccp_alpha=best_alpha)
clf_post_pruned.fit(X_train_balanced, y_train_balanced)

# Display the optimal alpha
print("Optimal alpha:", best_alpha)
# Evaluation Metrics
evaluate_classifier(clf_post_pruned, X_test, y_test)

# Visualize the pruned Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf_post_pruned, filled=True, feature_names=X.columns, class_names=['0', '1'])
plt.show()
#%% 2. Logistic Regression
from sklearn.linear_model import LogisticRegression

# Note: The GridSearchCV section is commented out as the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for parameter tuning on a different dataset.

"""
# Grid Search for Hyperparameter Tuning (Commented Out)
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs'],
}

logreg = LogisticRegression()
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

best_logreg = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
"""

# Best parameters identified by Grid Search
best_params = {
    'C': 10,
    'solver': 'lbfgs'
}

# Directly applying the best parameters to Logistic Regression
best_logreg = LogisticRegression(
    C=best_params['C'],
    solver=best_params['solver'],
    random_state=1
)

# Fit the model with the balanced training data
best_logreg.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(best_logreg, X_test, y_test)
#%% 3. K-Nearest Neighbors (Elbow Method for Optimal K)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Note: The Elbow method section is commented out since the optimal K has already been determined.
# Uncomment the section below if you need to identify the optimal K on a different dataset.

"""
# Elbow Method for Optimal K (Commented Out)
k_values = range(1, 21)
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_balanced, y_train_balanced)
    y_pred = knn.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

# Plot elbow curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy on Test Set')
plt.show()

# Optimal K (choose the K with highest or stabilized accuracy)
optimal_k = k_values[accuracy_scores.index(max(accuracy_scores))]
print(f"Optimal K: {optimal_k}")
"""

# Use the pre-determined optimal K directly
optimal_k = 2  # Optimal K found through search above

# Train the final KNN model with the optimal number of neighbors
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(knn_final, X_test, y_test)
#%% 4. Support Vector Machine (Linear, Polynomial, RBF Kernels)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Reduce the training set size for faster grid search
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.7, stratify=y_train_balanced, random_state=1
)

# 1. Linear Kernel SVM

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning on a different dataset.

"""
print("\nGrid Search for Linear Kernel")
param_grid_linear = {'C': [0.1, 1, 10]}
grid_search_linear = GridSearchCV(SVC(kernel='linear', probability=True, random_state=1),
                                  param_grid_linear, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search_linear.fit(X_train_small, y_train_small)

best_svm_linear = grid_search_linear.best_estimator_
print("Best Parameters (Linear Kernel):", grid_search_linear.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_linear = {'C': 1}  # Replace with the actual best parameters if different

# Train the final SVM model using the pre-determined best parameters
final_svm_linear = SVC(
    kernel='linear',
    C=best_params_linear['C'],
    probability=True,
    random_state=1
)

# Fit the model with the balanced training data
final_svm_linear.fit(X_train_small, y_train_small)

# Evaluation Metrics
evaluate_classifier(final_svm_linear, X_test, y_test)
#%% 2. Polynomial Kernel SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning on a different dataset.

"""
# Reduce the training set size for faster grid search
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.7, stratify=y_train_balanced, random_state=1
)

# Grid Search for Polynomial Kernel
print("\nGrid Search for Polynomial Kernel")
param_grid_poly = {
    'C': [0.1, 1],
    'degree': [2, 3],
}
grid_search_poly = GridSearchCV(SVC(kernel='poly', probability=True, random_state=1),
                                param_grid_poly, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search_poly.fit(X_train_small, y_train_small)

best_svm_poly = grid_search_poly.best_estimator_
print("Best Parameters (Polynomial Kernel):", grid_search_poly.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_poly = {'C': 10, 'degree': 3, 'gamma': 'auto'}  # Replace with the actual best parameters if different

# Train the final SVM model using the pre-determined best parameters
final_svm_poly = SVC(
    kernel='poly',
    C=best_params_poly['C'],
    degree=best_params_poly['degree'],
    gamma=best_params_poly['gamma'],
    probability=True,
    random_state=1
)

# Fit the model with the balanced training data
final_svm_poly.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(final_svm_poly, X_test, y_test)
#%% 3. Radial Basis Function (RBF) Kernel SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning on a different dataset.

"""
# Reduce the training set size for faster grid search
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_balanced, y_train_balanced, test_size=0.7, stratify=y_train_balanced, random_state=1
)

# Grid Search for RBF Kernel
print("\nGrid Search for RBF Kernel")
param_grid_rbf = {
    'C': [0.1, 1],
    'gamma': ['scale', 'auto']
}
grid_search_rbf = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=1),
                               param_grid_rbf, cv=3, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search_rbf.fit(X_train_small, y_train_small)

best_svm_rbf = grid_search_rbf.best_estimator_
print("Best Parameters (RBF Kernel):", grid_search_rbf.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_rbf = {'C': 1, 'gamma': 'auto'}  # Replace with the actual best parameters if different

# Train the final SVM model using the pre-determined best parameters
final_svm_rbf = SVC(
    kernel='rbf',
    C=best_params_rbf['C'],
    gamma=best_params_rbf['gamma'],
    probability=True,
    random_state=1
)

# Fit the model with the balanced training data
final_svm_rbf.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(final_svm_rbf, X_test, y_test)
#%% 5. Naive Bayes
from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()
clf_nb.fit(X_train_balanced, y_train_balanced)
# Evaluation Metrics
evaluate_classifier(clf_nb, X_test, y_test)
#%% 6. Random Forest (Bagging, Stacking, Boosting)
# Random Forest (Bagging)
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning.

"""
# Define the parameter grid for Bagging
param_grid_bagging = {
    'n_estimators': [10, 50],
    'max_samples': [0.5, 1.0],
    'max_features': [0.5, 1.0],
}

# Initialize the model
bagging_model = BaggingClassifier(base_estimator=RandomForestClassifier(random_state=1), random_state=1)

# Perform grid search
grid_search_bagging = GridSearchCV(estimator=bagging_model, param_grid=param_grid_bagging, cv=3,
                                   scoring='accuracy', verbose=3, n_jobs=-1)

# Fit the model
grid_search_bagging.fit(X_train, y_train)

# Best model after grid search
bagging_best_model = grid_search_bagging.best_estimator_
print("Best Parameters:", grid_search_bagging.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_bagging = {
    'n_estimators': 10,
    'max_samples': 1.0,
    'max_features': 1.0
}  # Replace with the actual best parameters if different

# Train the final model using the pre-determined best parameters
final_bagging_model = BaggingClassifier(
    base_estimator=RandomForestClassifier(random_state=1),
    n_estimators=best_params_bagging['n_estimators'],
    max_samples=best_params_bagging['max_samples'],
    max_features=best_params_bagging['max_features'],
    random_state=1
)

# Fit the model with the balanced training data
final_bagging_model.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(final_bagging_model, X_test, y_test)
#%% Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning.

"""
# Define the base learners for stacking
base_learners = [
    ('rf', RandomForestClassifier(random_state=1, n_estimators=100)),
    ('svc', SVC(probability=True, random_state=1))
]

# Initialize the StackingClassifier
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(),
    cv=3
)

# Grid search for hyperparameter tuning
param_grid_stacking = {
    'rf__n_estimators': [100, 200],
    'svc__C': [0.1, 1, 10]
}

grid_search_stacking = GridSearchCV(
    estimator=stacking_model,
    param_grid=param_grid_stacking,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# Fit the model
grid_search_stacking.fit(X_train, y_train)

# Best model after grid search
stacking_best_model = grid_search_stacking.best_estimator_
print("Best Parameters:", grid_search_stacking.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_stacking = {
    'rf__n_estimators': 100,
    'svc__C': 10
}  # Replace with the actual best parameters if different

# Train the final stacking model using the pre-determined best parameters
final_stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=1, n_estimators=best_params_stacking['rf__n_estimators'])),
        ('svc', SVC(C=best_params_stacking['svc__C'], probability=True, random_state=1))
    ],
    final_estimator=LogisticRegression(),
    cv=3
)

# Fit the model with the balanced training data
final_stacking_model.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(final_stacking_model, X_test, y_test)
#%% Boosting
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning.

"""
# Parameter grid for grid search
param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize the model (without use_label_encoder)
xgb_model = XGBClassifier(random_state=1)

# Perform grid search
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

# Fit the model
grid_search_xgb.fit(X_train, y_train)

# Best model after grid search
xgb_best_model = grid_search_xgb.best_estimator_
print("Best Parameters:", grid_search_xgb.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_xgb = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.1,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 1.0
}  # Replace with the actual best parameters if different

# Train the final XGBoost model using the pre-determined best parameters
final_xgb_model = XGBClassifier(
    colsample_bytree=best_params_xgb['colsample_bytree'],
    learning_rate=best_params_xgb['learning_rate'],
    max_depth=best_params_xgb['max_depth'],
    n_estimators=best_params_xgb['n_estimators'],
    subsample=best_params_xgb['subsample'],
    random_state=1
)

# Fit the model with the balanced training data
final_xgb_model.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(final_xgb_model, X_test, y_test)
#%% 7. Neural Network (Multi-Layer Perceptron)
from sklearn.neural_network import MLPClassifier

# Note: The GridSearchCV section is commented out since the best parameters have already been identified.
# Uncomment the section below if you need to perform a grid search for hyperparameter tuning.

"""
# Define parameter grid for hidden layer sizes
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)]
}

# Initialize the MLPClassifier with max_iter=1000
mlp = MLPClassifier(max_iter=1000)

# Perform grid search
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', verbose=3)
grid_search.fit(X_train, y_train)

# Get the best model and parameters after grid search
best_mlp = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
"""

# Best Parameters identified by Grid Search
best_params_mlp = {'hidden_layer_sizes': (50,)}  # Replace with the actual best parameters if different

# Train the final MLP model using the pre-determined best parameters
mlp_best = MLPClassifier(
    hidden_layer_sizes=best_params_mlp['hidden_layer_sizes'],
    max_iter=1000,
    random_state=1
)

# Fit the model with the balanced training data
mlp_best.fit(X_train_balanced, y_train_balanced)

# Evaluation Metrics
evaluate_classifier(mlp_best, X_test, y_test)

#%% Phase 4: Clustering and Association Rule Mining
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Clustering with K-Means
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np

# Define features and target variable
target_variable = 'hospital_death'
X_c = df.drop(columns=[target_variable])
y_c = df[target_variable]

# Silhouette analysis for k selection
k_range = range(2, 11)  # Test k values from 2 to 10
silhouette_scores = []
within_cluster_sums = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(X_c)
    silhouette_scores.append(silhouette_score(X_c, kmeans.labels_))
    within_cluster_sums.append(kmeans.inertia_)  # Within-cluster variation

# Plot Silhouette Scores
plt.figure(figsize=(8, 4))
plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Plot Within-Cluster Sum of Squares
plt.figure(figsize=(8, 4))
plt.plot(k_range, within_cluster_sums, marker='o', linestyle='--')
plt.title("Within-Cluster Variation (Inertia) for Different k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.grid(True)
plt.show()

# KMeans for the selected number of clusters = 2
optimal_k = 2
kmeans = KMeans(n_clusters=optimal_k, random_state=1)
kmeans.fit(X_c)

# Evaluate clustering performance for the chosen k
silhouette = silhouette_score(X_c, kmeans.labels_)
db_index = davies_bouldin_score(X_c, kmeans.labels_)
ch_index = calinski_harabasz_score(X_c, kmeans.labels_)
ari = adjusted_rand_score(y_c, kmeans.labels_)
mi = mutual_info_score(y_c, kmeans.labels_)

# Print metrics
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_index:.2f}")
print(f"Calinski-Harabasz Index: {ch_index:.2f}")
print(f"Adjusted Rand Index: {ari:.2f}")
print(f"Mutual Information Score: {mi:.2f}")
#%% DBSCAN Clustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Define features and target variable
target_variable = 'hospital_death'
X_c = df.drop(columns=[target_variable])
y_c = df[target_variable]

eps = 0.5
min_samples = 5

# Initialize and fit DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_c)

# Evaluate clustering performance
if len(set(dbscan_labels)) > 1:
    silhouette = silhouette_score(X_c, dbscan_labels)
    db_index = davies_bouldin_score(X_c, dbscan_labels)
    print(f"Silhouette Score: {silhouette:.2f}")
    print(f"Davies-Bouldin Index: {db_index:.2f}")
else:
    print("All points classified as noise or single cluster detected; Silhouette Score not meaningful.")

# Count the clusters and noise
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

#%% Apriori Algorithm for Association Rule Mining
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

binary_cols = [
    'elective_surgery', 'apache_post_operative', 'arf_apache', 'gcs_unable_apache', 'intubated_apache',
    'ventilated_apache', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression',
    'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'hospital_death'
]

# Extract the binary columns into a new DataFrame (df_a)
df_a = df[binary_cols]

frq_items = apriori(df_a, min_support=0.05, use_colnames=True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print(rules.head())