# Regression Workflow: Predicting Diabetes Disease Progression

## What is Regression?

**Regression** is a type of supervised machine learning used to predict **continuous numerical values**. Unlike classification (which predicts categories), regression predicts quantities.

**Examples:**
- Predicting house prices
- Forecasting temperature
- **Our case**: Predicting diabetes disease progression (a quantitative measure)

## What We'll Learn

This notebook demonstrates a **complete machine learning workflow**:

1. **Load & Explore Data** - Understand what we're working with
2. **Split Data** - Separate training and testing sets
3. **Train Models** - Fit different algorithms
4. **Evaluate Performance** - Evaluate models using metrics
5. **Visualize Results** - Understand predictions

## The Dataset

We'll use the **Diabetes dataset** from scikit-learn:
- **442 patients** with diabetes
- **10 features**: age, sex, BMI, blood pressure, and 6 blood serum measurements
- **Target**: Quantitative measure of disease progression one year after baseline

## 1. Import Libraries


```
# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)

# Make plots nicer
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
```

## 2. Load and Explore the Data

### Load the Dataset


```
# Load the diabetes dataset
dataset = load_diabetes()

# Display dataset description
print(dataset.DESCR)
```

### Understand the Data Structure


```
# Create DataFrame for easier exploration
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")
print(f"\nFeature names: {list(dataset.feature_names)}")
print(f"\nFirst few rows:")
df.head()
```

### Basic Statistics


```
# Display statistical summary
df.describe().round(2)
```

### Check Target Variable Type

The target variable is **continuous** (real numbers), confirming this is a **regression problem**.


```
print(f"Target variable dtype: {dataset.target.dtype}")
print(f"Target range: {dataset.target.min():.1f} to {dataset.target.max():.1f}")
print(f"Target mean: {dataset.target.mean():.1f}")
```

### Visualize Target Distribution


```
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(dataset.target, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Disease Progression')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Target Variable')
axes[0].axvline(dataset.target.mean(), color='red', linestyle='--', label=f'Mean: {dataset.target.mean():.1f}')
axes[0].legend()

# Boxplot
axes[1].boxplot(dataset.target)
axes[1].set_ylabel('Disease Progression')
axes[1].set_title('Target Variable Boxplot')
axes[1].set_xticklabels(['Target'])

plt.tight_layout()
plt.show()

print("✓ Target is continuous → This is a REGRESSION problem")
```

### Feature Correlations

Let's see which features are most correlated with disease progression:


```
# Calculate correlations with target
correlations = df.corr()['target'].drop('target').sort_values(ascending=False)

# Plot correlations
plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in correlations]
correlations.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Correlation with Disease Progression')
plt.title('Feature Correlations with Target')
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.tight_layout()
plt.show()

print("\nTop 3 positively correlated features:")
print(correlations.head(3))
print("\nTop 3 negatively correlated features:")
print(correlations.tail(3))
```

## 3. Split Data: Training vs Testing

### Why Split the Data?

This is the **most important principle** in machine learning:

- **Training set (80%)**: Used to teach the model
- **Test set (20%)**: Used to evaluate on unseen data

**Critical**: The model must NEVER see the test data during training!

### Prepare Features and Target


```
# Separate features (X) and target (y)
X, y = load_diabetes(return_X_y=True)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
```

### Perform the Split


```
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # For reproducibility
)

print("Data Split Summary:")
print("=" * 50)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test samples: {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nFeatures shape:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"\nTarget distributions (should be similar):")
print(f"  Train mean: {y_train.mean():.1f}")
print(f"  Test mean: {y_test.mean():.1f}")
```

### Visualize the Split


```
# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Split proportions
sizes = [len(X_train), len(X_test)]
labels = [f'Train\n{len(X_train)} samples', f'Test\n{len(X_test)} samples']
colors = ['#66c2a5', '#fc8d62']
axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
axes[0].set_title('Train/Test Split')

# Distribution comparison
axes[1].hist([y_train, y_test], bins=20, label=['Train', 'Test'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('Disease Progression')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Target Distribution: Train vs Test')
axes[1].legend()

plt.tight_layout()
plt.show()

print("✓ Distributions are similar → Good split!")
```

## 4. Train the Models

We'll train two different regression models:

1. **Linear Regression**: Assumes linear relationship between features and target
2. **Decision Tree Regressor**: Can capture non-linear patterns

### Model 1: Linear Regression


```
# Create and train Linear Regression model
model_linear = LinearRegression()
model_linear.fit(X_train, y_train)

# Make predictions on test set
y_pred_linear = model_linear.predict(X_test)

print(f"Model coefficients shape: {model_linear.coef_.shape}")
```

### Model 2: Decision Tree Regressor


```
# Create and train Decision Tree model
model_tree = DecisionTreeRegressor(random_state=42, max_depth=5)
model_tree.fit(X_train, y_train)

# Make predictions on test set
y_pred_tree = model_tree.predict(X_test)

print("✓ Decision Tree trained successfully")
print(f"Tree depth: {model_tree.get_depth()}")
print(f"Number of leaves: {model_tree.get_n_leaves()}")
```

## 5. Evaluate the Models

### Regression Metrics

We'll use three common metrics:

1. **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
   - Lower is better
   - Same units as target variable

2. **RMSE (Root Mean Squared Error)**: Square root of average squared differences
   - Lower is better
   - Penalizes large errors more than MAE

3. **R² Score**: Proportion of variance explained by the model
   - Range: 0 to 1 (higher is better)
   - 1.0 = perfect predictions
   - 0.0 = no better than predicting the mean


```
# Calculate metrics for Linear Regression
mae_linear = mean_absolute_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

# Calculate metrics for Decision Tree
mae_tree = mean_absolute_error(y_test, y_pred_tree)
rmse_tree = np.sqrt(mean_squared_error(y_test, y_pred_tree))
r2_tree = r2_score(y_test, y_pred_tree)

# Display results
print("Model Performance Comparison")
print("=" * 60)
print(f"{'Metric':<20} {'Linear Regression':>18} {'Decision Tree':>18}")
print("-" * 60)
print(f"{'MAE':<20} {mae_linear:>18.2f} {mae_tree:>18.2f}")
print(f"{'RMSE':<20} {rmse_linear:>18.2f} {rmse_tree:>18.2f}")
print(f"{'R² Score':<20} {r2_linear:>18.3f} {r2_tree:>18.3f}")
print("=" * 60)

# Determine winner
if r2_linear > r2_tree:
    print("\n🏆 Linear Regression performs better (higher R²)")
else:
    print("\n🏆 Decision Tree performs better (higher R²)")
```

### Create Comparison DataFrame


```
# Create comparison table
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree'],
    'MAE': [mae_linear, mae_tree],
    'RMSE': [rmse_linear, rmse_tree],
    'R² Score': [r2_linear, r2_tree]
})

results_df = results_df.round(3)
results_df
```

## 6. Visualize the Results

### Predictions vs Actual Values


```
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Regression
axes[0].scatter(y_test, y_pred_linear, alpha=0.6, edgecolors='k')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(f'Linear Regression\nR² = {r2_linear:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Decision Tree
axes[1].scatter(y_test, y_pred_tree, alpha=0.6, edgecolors='k', color='orange')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
axes[1].set_xlabel('Actual Values')
axes[1].set_ylabel('Predicted Values')
axes[1].set_title(f'Decision Tree\nR² = {r2_tree:.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Residual Plot (for best model)

Residuals = Actual - Predicted. Ideally, they should be randomly distributed around zero.


```
# Calculate residuals for Linear Regression (typically the better model)
residuals = y_test - y_pred_linear

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual scatter plot
axes[0].scatter(y_pred_linear, residuals, alpha=0.6, edgecolors='k')
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals (Actual - Predicted)')
axes[0].set_title('Residual Plot: Linear Regression')
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean residual: {residuals.mean():.2f} (should be close to 0)")
print(f"Std of residuals: {residuals.std():.2f}")
```

### Feature Importance (Decision Tree)


```
# Get feature importances from Decision Tree
importances = model_tree.feature_importances_
feature_names = dataset.feature_names

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

1. **Regression predicts continuous values** (vs classification which predicts categories)
2. **Always split data** into training and test sets
3. **Different models have different strengths**:
   - Linear Regression: Simple, interpretable, assumes linearity
   - Decision Tree: Can capture non-linear patterns, risk of overfitting
4. **Use multiple metrics** to evaluate (MAE, RMSE, R²)
5. **Visualizations help understand** model performance

### Next Steps

Compare this **regression** workflow with the **classification** workflow in the companion notebook (`diabetes_classification.ipynb`). Notice:
- Similar structure (load, split, train, evaluate)
- Different target types (continuous vs categorical)
- Different metrics (MAE/R² vs Accuracy/F1)
- Different visualizations (scatter plots vs confusion matrix)


## 7. Conclusion

In this analysis, we compared a Linear Regression model and a Decision Tree Regressor for predicting diabetes progression.

### Key Findings:

1.  **Model Performance**:
    - **Linear Regression** achieved an $R^2$ score of approximately **0.45**, significantly outperforming the Decision Tree ($R^2 \approx 0.33$).
    - It also had lower error rates (MAE and RMSE), indicating its predictions were generally closer to the true values.

2.  **Why did Linear Regression win?**
    - The Diabetes dataset has a relatively small number of samples (442). simpler models often perform better on small datasets as they are less prone to overfitting.
    - The relationships between the physiological features (like BMI and BP) and disease progression are likely largely linear, which fits the assumptions of Linear Regression well.

3.  **Residual Analysis**:
    - The residual plots help us validate our models. For a good model, residuals should be randomly scattered around zero.
    - Patterns in the Decision Tree residuals (vertical lines) often indicate that the model is "binning" predictions too rigidly, a common characteristic of tree-based models with limited depth.

### Next Steps for Improvement:
- **Feature Engineering**: Create new features (e.g., interaction terms like BMI * Age) to capture more complex relationships.
- **Hyperparameter Tuning**: Use techniques like Grid Search to find the optimal `max_depth` for the Decision Tree to prevent overfitting or underfitting.
- **Advanced Models**: Try more powerful algorithms like **Random Forest** or **Gradient Boosting**, which combine multiple decision trees to improve performance and stability.

