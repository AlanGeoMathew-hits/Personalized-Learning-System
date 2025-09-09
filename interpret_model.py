import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the processed dataset
df = pd.read_csv('processed_student_data.csv')

# --- Retrain the model to access its properties ---
X = df.drop('Risk_Category', axis=1)
y = df['Risk_Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# --- 1. Analyze Feature Importance ---

# Get feature importances from the trained model
importances = rf_classifier.feature_importances_
feature_names = X.columns

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance_df.head(10))

# Visualize the feature importances (Corrected line)
plt.figure(figsize=(12, 8))
# This version fixes the warning and the error. It will produce a legend, which is acceptable.
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), hue='Feature', palette='viridis')
plt.title('Top 15 Most Important Features for Risk Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nFeature importance plot saved as 'feature_importance.png'")


# --- 2. Design the Feedback Logic ---

def generate_feedback(student_data, prediction):
    """
    Generates personalized feedback for a student based on their data and prediction.
    - student_data: A pandas Series containing a single student's feature values.
    - prediction: The model's prediction (1 for 'at-risk', 0 for 'not at-risk').
    """
    if prediction == 0:
        return "Prediction: Not At-Risk. Keep up the great work! Your study habits and engagement are leading to positive results."

    feedback = "Prediction: At-Risk. Here are some areas to focus on:\n"

    # Check the top 3 most important features from our analysis
    if student_data['Previous_Scores'] < 75:
        feedback += f"- Your previous scores (currently {student_data['Previous_Scores']}) are a key factor. Focusing on foundational topics could significantly boost your performance.\n"

    if student_data['Hours_Studied'] < 20:
        feedback += f"- Your study time (currently {student_data['Hours_Studied']} hours) is lower than average. Let's try to build a consistent study schedule.\n"

    if student_data['Attendance'] < 80:
        feedback += f"- Your attendance (currently {student_data['Attendance']}%) is a bit low. Attending classes regularly is one of the best ways to stay on track.\n"

    # Add feedback for a categorical feature
    if student_data['Motivation_Level_Low'] == 1:
        feedback += "- It seems your motivation level is low. Let's connect with a counselor to set some exciting academic goals.\n"

    if len(feedback) < 100:  # If no specific feedback was triggered
        feedback += "- The model indicates a potential risk. It would be beneficial to meet with a teacher or advisor to review your study plan."

    return feedback


# --- Demonstrate the feedback system ---

# Find an example of an at-risk student from the test set
X_test_with_pred = X_test.copy()
X_test_with_pred['prediction'] = rf_classifier.predict(X_test)
X_test_with_pred['actual'] = y_test

# Get the first student the model correctly predicted as "at-risk"
at_risk_student = X_test_with_pred[(X_test_with_pred['prediction'] == 1) & (X_test_with_pred['actual'] == 1)].iloc[0]

print("\n--- Example Feedback Generation ---")
print("Analyzing a sample 'at-risk' student:")
print("Student's Data (subset):")
print(at_risk_student[['Previous_Scores', 'Hours_Studied', 'Attendance', 'Motivation_Level_Low']])
print("\nGenerated Feedback:")
feedback_message = generate_feedback(at_risk_student, at_risk_student['prediction'])
print(feedback_message)