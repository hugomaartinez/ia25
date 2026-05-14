# Classification Workflow: Predicting Diabetes Presence

## What is Classification?

**Classification** is a type of supervised machine learning used to predict **categorical labels** (discrete classes). Unlike regression (which predicts continuous values), classification assigns observations to predefined categories.

**Examples:**
- Email spam detection (spam vs not spam)
- Image recognition (cat vs dog)
- **Our case**: Diabetes diagnosis (has diabetes vs no diabetes)

## Regression vs Classification: Same Domain, Different Problems

This notebook uses diabetes data, just like the regression notebook, but solves a different problem:

| Aspect | Regression Notebook | This Classification Notebook |
|--------|---------------------|------------------------------|
| **Question** | How much will disease progress? | Does patient have diabetes? |
| **Target type** | Continuous (25-346) | Binary (0 or 1) |
| **Problem type** | Regression | Classification |

## What We'll Learn

This notebook demonstrates the **same ML workflow** as regression, but for classification:

1. **Load & Explore Data** - Understand the features and target classes
2. **Split Data** - Separate training and testing sets
3. **Train Models** - Fit different classification algorithms
4. **Evaluate Performance** - Compare models using classification metrics
5. **Visualize Results** - Understand predictions with confusion matrices

## The Dataset

We'll use the **Pima Indians Diabetes Database**:
- **768 female patients** of Pima Indian heritage
- **8 features**: pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, pedigree function, age
- **Target**: Binary outcome (0 = no diabetes, 1 = has diabetes)

## 1. Import Libraries


```
# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data download
import kagglehub

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    ConfusionMatrixDisplay
)

# Set random seed for reproducibility
np.random.seed(42)

# Make plots nicer
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## 2. Load and Explore the Data

### Download the Dataset


```
# Download latest version from Kaggle
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
print("Path to dataset files:", path)
```

### Load the Data


```
# Load the CSV file
import os
csv_path = os.path.join(path, 'diabetes.csv')
df = pd.read_csv(csv_path)

print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nFirst few rows:")
df.head()
```

### Feature Descriptions

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (2 hours in oral glucose tolerance test)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (genetic influence)
- **Age**: Age in years
- **Outcome**: Class variable (0 = no diabetes, 1 = has diabetes)

### Data Info


```
df.info()
```

### Basic Statistics


```
df.describe().round(2)
```

### Check Target Variable Type

The target variable is **binary** (0 or 1), confirming this is a **classification problem**.


```
print(f"Target variable (Outcome) unique values: {df['Outcome'].unique()}")
print(f"Target variable dtype: {df['Outcome'].dtype}")
print(f"\nClass distribution:")
print(df['Outcome'].value_counts().sort_index())
print(f"\nClass balance:")
print(df['Outcome'].value_counts(normalize=True).sort_index())
```

### Visualize Class Distribution


```
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
class_counts = df['Outcome'].value_counts().sort_index()
axes[0].bar(['No Diabetes', 'Has Diabetes'], class_counts.values, 
            color=['#66c2a5', '#fc8d62'], edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title('Class Distribution')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(class_counts.values, labels=['No Diabetes', 'Has Diabetes'], 
            autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62'], startangle=90)
axes[1].set_title('Class Proportions')

plt.tight_layout()
plt.show()

print("✓ Target is binary (0/1) → This is a CLASSIFICATION problem")
```

## 3. Data Preprocessing

### Handling Missing Values

In this dataset, a value of **0** in features like Glucose, BloodPressure, SkinThickness, Insulin, and BMI indicates missing data (it's physically impossible to have 0 blood pressure or BMI).

We will handle this by:
1. Replacing `0` with `NaN` in these specific columns.
2. Imputing (filling) the `NaN` values with the **median** of the column. Median is more robust to outliers than mean.


```
# Features where 0 signifies missing data
features_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0 with NaN
df[features_with_zeros] = df[features_with_zeros].replace(0, np.nan)

# Check how many missing values we have now
print("Missing values (NaN) after replacing 0s:")
print(df[features_with_zeros].isnull().sum())

# Impute with median
for feature in features_with_zeros:
    median_val = df[feature].median()
    df[feature].fillna(median_val, inplace=True)
    print(f"Filled {feature} with median: {median_val:.1f}")

print("\n✓ Missing values handled!")
```

### Feature Distributions by Class


```
# Select a few key features to visualize
key_features = ['Glucose', 'BMI', 'Age', 'Pregnancies']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(key_features):
    # Separate data by class
    no_diabetes = df[df['Outcome'] == 0][feature]
    has_diabetes = df[df['Outcome'] == 1][feature]
    
    axes[idx].hist([no_diabetes, has_diabetes], bins=20, label=['No Diabetes', 'Has Diabetes'],
                   color=['#66c2a5', '#fc8d62'], alpha=0.7, edgecolor='black')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_title(f'{feature} Distribution by Class')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Correlation Analysis


```
# Calculate correlations with outcome
correlations = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in correlations]
correlations.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Correlation with Diabetes')
plt.title('Feature Correlations with Target')
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.show()

print("\nTop 3 features most correlated with diabetes:")
print(correlations.head(3))
```

## 3. Split Data: Training vs Testing

### Why Split the Data?

Same principle as regression:
- **Training set (80%)**: Teach the model
- **Test set (20%)**: Evaluate on unseen data

**Additional for classification**: We use `stratify` to maintain class balance in both sets.

### Prepare Features and Target


```
# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature names: {list(X.columns)}")
```

### Perform the Split (with Stratification)


```
# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class proportions
)

print("Data Split Summary:")
print("=" * 50)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

print(f"\nClass balance in training set:")
print(y_train.value_counts(normalize=True).sort_index())
print(f"\nClass balance in test set:")
print(y_test.value_counts(normalize=True).sort_index())
print("\n✓ Class proportions maintained!")
```

### Visualize the Split


```
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Split proportions
sizes = [len(X_train), len(X_test)]
labels = [f'Train\n{len(X_train)} samples', f'Test\n{len(X_test)} samples']
colors = ['#66c2a5', '#fc8d62']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
axes[0].set_title('Train/Test Split')

# Class distribution comparison
train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()
x = np.arange(2)
width = 0.35
axes[1].bar(x - width/2, train_counts.values, width, label='Train', edgecolor='black')
axes[1].bar(x + width/2, test_counts.values, width, label='Test', edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('Class Distribution: Train vs Test')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['No Diabetes', 'Has Diabetes'])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

## 4. Feature Scaling

### Why Scale?

Logistic Regression (and many algorithms) perform better with scaled features because:
- Features have different ranges (e.g., Age: 20-80, BMI: 0-67)
- Scaling ensures all features contribute equally
- Improves convergence speed

**StandardScaler** transforms features to have mean=0 and std=1.


```
# Create and fit scaler on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform both sets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Complete!")
print(f"\nBefore scaling:")
print(f"  Feature ranges vary (e.g., Age: {X_train['Age'].min():.0f}-{X_train['Age'].max():.0f}, "
      f"Glucose: {X_train['Glucose'].min():.0f}-{X_train['Glucose'].max():.0f})")
print(f"\nAfter scaling:")
print(f"  All features have mean ≈ 0, std ≈ 1")
print(f"  Mean: {X_train_scaled.mean():.6f}")
print(f"  Std: {X_train_scaled.std():.6f}")
```

## 5. Train the Models

We'll train two classification models:

1. **Logistic Regression**: Linear classifier (analogous to Linear Regression)
2. **Decision Tree Classifier**: Non-linear classifier (analogous to Decision Tree Regressor)

### Model 1: Logistic Regression


```
# Create and train Logistic Regression model
model_logistic = LogisticRegression(max_iter=1000, random_state=42)
model_logistic.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred_logistic = model_logistic.predict(X_test_scaled)
```

### Model 2: Decision Tree Classifier


```
# Create and train Decision Tree model (no scaling needed, but we'll use scaled for fair comparison)
model_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
model_tree.fit(X_train_scaled, y_train)

# Make predictions on test set
y_pred_tree = model_tree.predict(X_test_scaled)
print(f"Tree depth: {model_tree.get_depth()}")
print(f"Number of leaves: {model_tree.get_n_leaves()}")
```

## 6. Evaluate the Models

### Classification Metrics

Different from regression! We use:

1. **Accuracy**: Percentage of correct predictions
   - Range: 0 to 1 (higher is better)
   - Can be misleading with imbalanced classes

2. **Precision**: Of predicted positives, how many are truly positive?
   - Formula: TP / (TP + FP)

3. **Recall (Sensitivity)**: Of actual positives, how many did we catch?
   - Formula: TP / (TP + FN)

4. **F1-Score**: Harmonic mean of precision and recall
   - Balances precision and recall


```
# Calculate accuracy for both models
acc_logistic = accuracy_score(y_test, y_pred_logistic)
acc_tree = accuracy_score(y_test, y_pred_tree)

print("Model Performance Comparison")
print("=" * 60)
print(f"{'Model':<25} {'Accuracy':>15}")
print("-" * 60)
print(f"{'Logistic Regression':<25} {acc_logistic:>15.3f}")
print(f"{'Decision Tree':<25} {acc_tree:>15.3f}")
print("=" * 60)

# Determine winner
if acc_logistic > acc_tree:
    print("\n🏆 Logistic Regression performs better")
else:
    print("\n🏆 Decision Tree performs better")
```

### Detailed Classification Reports


```
print("Logistic Regression - Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred_logistic, 
                          target_names=['No Diabetes', 'Has Diabetes']))

print("\nDecision Tree - Classification Report:")
print("=" * 60)
print(classification_report(y_test, y_pred_tree,
                          target_names=['No Diabetes', 'Has Diabetes']))
```

### Create Comparison DataFrame


```
# Extract metrics from classification reports
from sklearn.metrics import precision_score, recall_score, f1_score

results_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [acc_logistic, acc_tree],
    'Precision': [
        precision_score(y_test, y_pred_logistic, average='weighted'),
        precision_score(y_test, y_pred_tree, average='weighted')
    ],
    'Recall': [
        recall_score(y_test, y_pred_logistic, average='weighted'),
        recall_score(y_test, y_pred_tree, average='weighted')
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_logistic, average='weighted'),
        f1_score(y_test, y_pred_tree, average='weighted')
    ]
})

results_df = results_df.round(3)
results_df
```

## 7. Visualize the Results

### Confusion Matrices

A confusion matrix shows:
- **True Positives (TP)**: Correctly predicted diabetes
- **True Negatives (TN)**: Correctly predicted no diabetes
- **False Positives (FP)**: Predicted diabetes, but patient doesn't have it
- **False Negatives (FN)**: Predicted no diabetes, but patient has it


```
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression confusion matrix
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_logistic, 
                                display_labels=['No Diabetes', 'Has Diabetes'])
disp1.plot(ax=axes[0], cmap='Blues', values_format='d')
axes[0].set_title(f'Logistic Regression\nAccuracy = {acc_logistic:.3f}')

# Decision Tree confusion matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_tree,
                                display_labels=['No Diabetes', 'Has Diabetes'])
disp2.plot(ax=axes[1], cmap='Oranges', values_format='d')
axes[1].set_title(f'Decision Tree\nAccuracy = {acc_tree:.3f}')

plt.tight_layout()
plt.show()
```

### Feature Importance (Decision Tree)


```
# Get feature importances
importances = model_tree.feature_importances_
feature_names = X.columns

# Create DataFrame and sort
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], edgecolor='black')
plt.xlabel('Importance')
plt.title('Feature Importance (Decision Tree)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\nTop 3 most important features:")
print(importance_df.head(3).to_string(index=False))
```

## Summary

### What We Learned

1. **Classification predicts categories** (vs regression which predicts continuous values)
2. **Same domain, different problem**: Both notebooks use diabetes data
   - Regression: Predict disease progression (how much?)
   - Classification: Predict diabetes presence (yes or no?)
3. **Workflow is similar**:
   - Load data → Split → Scale → Train → Evaluate
4. **Different metrics**:
   - Regression: MAE, RMSE, R²
   - Classification: Accuracy, Precision, Recall, F1
5. **Different visualizations**:
   - Regression: Scatter plots, residuals
   - Classification: Confusion matrix
6. **Stratification matters** in classification to maintain class balance

### Comparison with Regression Notebook

| Aspect | Regression | Classification |
|--------|-----------|----------------|
| **Dataset** | UCI Diabetes | Pima Indians |
| **Target** | Disease progression (continuous) | Has diabetes (binary) |
| **Question** | How much? | Which class? |
| **Models** | LinearRegression, DecisionTreeRegressor | LogisticRegression, DecisionTreeClassifier |
| **Metrics** | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |
| **Visualizations** | Scatter plots, residuals | Confusion matrix |
| **Scaling** | Not required for tree models | Important for LogisticRegression |

### Key Takeaway

The **ML workflow is the same**, but:
- The **problem type** determines whether to use regression or classification
- The **metrics and visualizations** differ based on the target variable type
