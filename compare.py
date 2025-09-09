import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# --- 1. Load and Prepare the Data ---
try:
    df = pd.read_csv('StudentPerformanceFactors.csv')
except FileNotFoundError:
    print("Error: 'StudentPerformanceFactors.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# Preprocessing steps from your previous work
# Handle missing values
for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    # Check if the column exists before trying to fill it
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Define the "Risk" Category
risk_threshold = df['Exam_Score'].quantile(0.25)
df['Risk_Category'] = (df['Exam_Score'] <= risk_threshold).astype(int)

# One-hot encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Prepare final X and y
X = df_processed.drop(['Exam_Score', 'Risk_Category'], axis=1)
y = df_processed['Risk_Category']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("--- Data Loaded and Prepared Successfully ---\n")


# --- 2. Train and Evaluate RandomForestClassifier ---
print("--- Training RandomForestClassifier ---")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)

print("\nPerformance Report for RandomForestClassifier:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(classification_report(y_test, y_pred_rf, target_names=['Not At-Risk', 'At-Risk']))


# --- 3. Train and Evaluate LogisticRegression ---
print("\n--- Training LogisticRegression ---")
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train, y_train)
y_pred_lr = lr_classifier.predict(X_test)

print("\nPerformance Report for LogisticRegression:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr) * 100:.2f}%")
print(classification_report(y_test, y_pred_lr, target_names=['Not At-Risk', 'At-Risk']))


# --- 4. Compare Models and Visualize ---
print("\n--- Model Comparison Summary ---")

# Calculate metrics for the 'At-Risk' class (label=1)
rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf)
}

lr_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
}

print("\nMetrics for predicting the 'At-Risk' class:")
print(f"{'Metric':<12} | {'RandomForest':<15} | {'LogisticRegression':<20}")
print("-" * 55)
for metric in rf_metrics:
    print(f"{metric:<12} | {rf_metrics[metric]:<15.3f} | {lr_metrics[metric]:<20.3f}")

# Create a DataFrame for easy plotting
comparison_df = pd.DataFrame({
    'RandomForest': rf_metrics,
    'LogisticRegression': lr_metrics
}).T # Transpose for plotting

# Visualize the comparison
plt.style.use('seaborn-v0_8-darkgrid')
comparison_df.plot(kind='bar', figsize=(12, 7))
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Metrics')
plt.ylim(0, 1.0) # Set y-axis from 0 to 1 for scores
plt.tight_layout()
plt.savefig('model_comparison.png')
print("\n✅ Performance comparison chart saved as 'model_comparison.png'")