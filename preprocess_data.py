import pandas as pd

# Load the cleaned dataset from the previous step
df = pd.read_csv('cleaned_student_data.csv')

# --- 1. Define the "Risk" Category ---

# Calculate the 25th percentile for Exam_Score
risk_threshold = df['Exam_Score'].quantile(0.25)
print(f"The risk threshold (25th percentile of Exam_Score) is: {risk_threshold}")

# Create the target variable 'Risk_Category'
# 1 for at-risk (score <= threshold), 0 for not at-risk
df['Risk_Category'] = (df['Exam_Score'] <= risk_threshold).astype(int)

# Check the distribution of the new category
print("\nDistribution of Risk Category:")
print(df['Risk_Category'].value_counts())


# --- 2. Encode Categorical Variables ---

# Identify categorical columns (those with dtype 'object')
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"\nCategorical columns to be one-hot encoded: {list(categorical_cols)}")

# Apply one-hot encoding
df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# The 'Exam_Score' column is no longer needed as we have our target 'Risk_Category'
df_processed = df_processed.drop('Exam_Score', axis=1)


# --- 3. Final Inspection and Save ---

# Display the first few rows of the fully processed dataframe
print("\nFirst 5 rows of the processed data:")
print(df_processed.head())

# Display the new structure of the dataframe
print("\nProcessed Dataset Info:")
df_processed.info()

# Save the processed data for the modeling step
df_processed.to_csv('processed_student_data.csv', index=False)
print("\nFully processed data saved to 'processed_student_data.csv'")