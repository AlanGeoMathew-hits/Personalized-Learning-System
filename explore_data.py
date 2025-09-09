import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

# --- 1. Handle Missing Values (Corrected Method) ---
# For categorical columns, we'll fill missing values with the mode.
for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    mode_val = df[col].mode()[0]
    # Corrected line: Explicitly assign the filled series back to the DataFrame column
    df[col] = df[col].fillna(mode_val)

print("Missing values after handling:")
print(df.isnull().sum())


# --- 2. Data Visualization ---

# Set the style for the plots
sns.set_style("whitegrid")

# a) Histogram of Exam Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['Exam_Score'], kde=True, bins=20)
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Frequency')
plt.savefig('exam_score_distribution.png')
# plt.show()

# b) Bar Chart: Average Exam Score by Motivation Level
plt.figure(figsize=(10, 6))
motivation_order = df.groupby('Motivation_Level')['Exam_Score'].mean().sort_values().index
sns.barplot(x='Motivation_Level', y='Exam_Score', data=df, order=motivation_order)
plt.title('Average Exam Score by Motivation Level')
plt.xlabel('Motivation Level')
plt.ylabel('Average Exam Score')
plt.savefig('avg_score_by_motivation.png')
# plt.show()

# c) Bar Chart: Average Exam Score by Parental Involvement
plt.figure(figsize=(10, 6))
parental_order = df.groupby('Parental_Involvement')['Exam_Score'].mean().sort_values().index
sns.barplot(x='Parental_Involvement', y='Exam_Score', data=df, order=parental_order)
plt.title('Average Exam Score by Parental Involvement')
plt.xlabel('Parental Involvement')
plt.ylabel('Average Exam Score')
plt.savefig('avg_score_by_parental_involvement.png')
# plt.show()

# d) Scatter Plot: Hours Studied vs. Exam Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=df)
plt.title('Hours Studied vs. Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.savefig('hours_studied_vs_score.png')
# plt.show()

# e) Correlation Heatmap for numerical features
plt.figure(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numerical_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.savefig('correlation_heatmap.png')
# plt.show()

# Save the cleaned data to a new CSV file
df.to_csv('cleaned_student_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_student_data.csv'")