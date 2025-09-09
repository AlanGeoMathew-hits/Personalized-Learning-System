import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- 1. Load Dataset ---
df = pd.read_csv('processed_student_data.csv')

X = df.drop('Risk_Category', axis=1)
y = df['Risk_Category']

# --- 2. Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Logistic Regression Model with Scaling ---
log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(
        max_iter=2000,
        solver='liblinear',       # better for small/medium datasets
        class_weight='balanced',  # fix class imbalance
        random_state=42
    ))
])

# --- 4. Train the Model ---
log_reg_pipeline.fit(X_train, y_train)

# --- 5. Evaluate Model ---
y_pred = log_reg_pipeline.predict(X_test)
print("\nClassification Report (Logistic Regression):")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nTraining Accuracy:", log_reg_pipeline.score(X_train, y_train))
print("Test Accuracy:", log_reg_pipeline.score(X_test, y_test))

# --- 6. Feature Importance (coefficients) ---
coefficients = log_reg_pipeline.named_steps['logreg'].coef_[0]
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute_Importance': abs(coefficients)
}).sort_values(by='Absolute_Importance', ascending=False)

print("\nTop 10 Most Important Features (by coefficient magnitude):")
print(feature_importance_df.head(10))

# --- 7. Visualize Feature Importance ---
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Absolute_Importance',
    y='Feature',
    data=feature_importance_df.head(15),
    hue='Feature',
    palette='viridis',
    dodge=False
)
plt.title('Top 15 Most Important Features (Logistic Regression)')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('logreg_feature_importance.png')
print("\nFeature importance plot saved as 'logreg_feature_importance.png'")

# --- 8. Feedback Generation ---
def generate_feedback(student_data, prediction):
    if prediction == 0:
        return "Prediction: Not At-Risk. Keep up the great work! Your study habits and engagement are leading to positive results."

    feedback = "Prediction: At-Risk. Here are some areas to focus on:\n"

    if student_data['Previous_Scores'] < 75:
        feedback += f"- Your previous scores (currently {student_data['Previous_Scores']}) are a key factor. Strengthening fundamentals could help improve performance.\n"

    if student_data['Hours_Studied'] < 20:
        feedback += f"- Your study time (currently {student_data['Hours_Studied']} hours) is relatively low. Creating a consistent study schedule can help.\n"

    if student_data['Attendance'] < 80:
        feedback += f"- Your attendance (currently {student_data['Attendance']}%) is a bit low. Regular class attendance can make a big difference.\n"

    if 'Motivation_Level_Low' in student_data and student_data['Motivation_Level_Low'] == 1:
        feedback += "- Your motivation seems low. Setting clear academic goals with guidance from a teacher or counselor may help.\n"

    if len(feedback) < 100:
        feedback += "- The model indicates a potential risk. Meeting with a teacher or advisor could be beneficial."

    return feedback

# --- 9. Demonstrate Feedback ---
X_test_with_pred = X_test.copy()
X_test_with_pred['prediction'] = y_pred
X_test_with_pred['actual'] = y_test

at_risk_students = X_test_with_pred[
    (X_test_with_pred['prediction'] == 1) & (X_test_with_pred['actual'] == 1)
]

if not at_risk_students.empty:
    at_risk_student = at_risk_students.iloc[0]
    print("\n--- Example Feedback Generation ---")
    print("Analyzing a sample 'at-risk' student:")
    print(at_risk_student[['Previous_Scores', 'Hours_Studied', 'Attendance', 'Motivation_Level_Low']])
    print("\nGenerated Feedback:")
    feedback_message = generate_feedback(at_risk_student, at_risk_student['prediction'])
    print(feedback_message)
else:
    print("\nNo correctly predicted at-risk students found in this test split.")

# --- 10. Save the Model ---
joblib.dump(log_reg_pipeline, 'student_risk_logreg_model.pkl')
print("\nModel saved as 'student_risk_logreg_model.pkl'")
