import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the processed dataset
try:
    df = pd.read_csv('processed_student_data.csv')
except FileNotFoundError:
    print("Make sure 'processed_student_data.csv' is in the same directory.")
    exit()

# --- 1. Split the Data ---

# 'X' contains all the feature columns (everything except the target)
X = df.drop('Risk_Category', axis=1)
# 'y' contains the target variable
y = df['Risk_Category']

# Split the data into 80% training and 20% testing sets
# random_state ensures that we get the same split every time we run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")


# --- 2. Train the Random Forest Model ---

# Initialize the classifier
# n_estimators is the number of trees in the forest
# random_state for reproducibility
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
print("\nTraining the Random Forest model...")
rf_classifier.fit(X_train, y_train)
print("Model training complete.")


# --- 3. Evaluate the Model ---

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

# Print a detailed classification report
print("\nClassification Report:")
# target_names helps label the report
print(classification_report(y_test, y_pred, target_names=['Not At-Risk (0)', 'At-Risk (1)']))

# Generate and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Not At-Risk', 'Predicted At-Risk'],
            yticklabels=['Actual Not At-Risk', 'Actual At-Risk'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

print("\nConfusion matrix plot saved as 'confusion_matrix.png'")