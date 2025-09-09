import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the final processed data
df = pd.read_csv('processed_student_data.csv')

# Separate features (X) and target (y)
X = df.drop('Risk_Category', axis=1)
y = df['Risk_Category']

# --- Train the model on the entire dataset ---
# We use all data now because this is our final production model
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training final model on all data...")
final_model.fit(X, y)
print("Model training complete.")

# --- Save the model and the columns ---
# Save the trained model to a file
joblib.dump(final_model, 'risk_predictor_model.joblib')

# Save the column list for the app to use
model_columns = list(X.columns)
joblib.dump(model_columns, 'model_columns.joblib')

print("\nModel and column list have been saved successfully.")
print("You can now run the Streamlit app!")