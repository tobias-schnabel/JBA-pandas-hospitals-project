import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 8)

general_df = pd.read_csv("test/general.csv")
prenatal_df = pd.read_csv("test/prenatal.csv")
sports_df = pd.read_csv("test/sports.csv")

general_columns = ['Unnamed: 0', 'hospital', 'gender', 'age', 'height', 'weight', 'bmi', 'diagnosis', 'blood_test',
                   'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']

# Change the column names
prenatal_df.columns = general_columns
sports_df.columns = general_columns

# Merge the DataFrames into one
merged_df = pd.concat([general_df, prenatal_df, sports_df], ignore_index=True)

# Delete the Unnamed: 0 column
merged_df = merged_df.drop(columns='Unnamed: 0')

# Step 1: Delete all empty rows
merged_df.dropna(how='all', inplace=True)

# Step 2: Correct gender values
gender_map = {'female': 'f', 'male': 'm', 'man': 'm', 'woman': 'f'}
merged_df['gender'] = merged_df['gender'].replace(gender_map)

# Step 3: Replace NaN in the gender column for prenatal
condition_prenatal = merged_df['hospital'] == 'prenatal'
merged_df.loc[condition_prenatal, 'gender'] = merged_df.loc[condition_prenatal, 'gender'].fillna('f')

# Step 4: Replace NaN in specified columns
columns_to_replace = ['bmi', 'diagnosis', 'blood_test', 'ecg', 'ultrasound', 'mri', 'xray', 'children', 'months']
merged_df[columns_to_replace] = merged_df[columns_to_replace].fillna(0)

# Print shape
# print(f"Data shape: {merged_df.shape}")

# # Print random 20 rows
# random_rows = merged_df.sample(n=20, random_state=30)
# print(random_rows)

hospital_count = merged_df['hospital'].value_counts()
max_hospital = hospital_count.idxmax()

general_stomach_issues = merged_df[(merged_df['hospital'] == 'general') & (merged_df['diagnosis'].str.contains('stomach', case=False, na=False))].shape[0]
general_total = merged_df[merged_df['hospital'] == 'general'].shape[0]
general_stomach_ratio = round(general_stomach_issues / general_total, 3)

sports_dislocation_issues = merged_df[(merged_df['hospital'] == 'sports') & (merged_df['diagnosis'].str.contains('dislocation', case=False, na=False))].shape[0]
sports_total = merged_df[merged_df['hospital'] == 'sports'].shape[0]
sports_dislocation_ratio = round(sports_dislocation_issues / sports_total, 3)

median_age_general = merged_df[merged_df['hospital'] == 'general']['age'].median()
median_age_sports = merged_df[merged_df['hospital'] == 'sports']['age'].median()
median_age_diff = abs(median_age_general - median_age_sports)

pivot_table_blood = pd.pivot_table(merged_df[merged_df['blood_test'] == 't'], values='blood_test', index=['hospital'], aggfunc='count')
most_blood_tests_hospital = pivot_table_blood.idxmax().iloc[0]
most_blood_tests_count = pivot_table_blood.max().iloc[0]


# print(f"The answer to the 1st question is {max_hospital}")
# print(f"The answer to the 2nd question is {general_stomach_ratio}")
# print(f"The answer to the 3rd question is {sports_dislocation_ratio}")
# print(f"The answer to the 4th question is {median_age_diff}")
# print(f"The answer to the 5th question is {most_blood_tests_hospital}, {most_blood_tests_count} blood tests")

# First plot
plt.hist(merged_df['age'].dropna(), bins=[0, 15, 35, 55, 70, 80], edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution Among All Hospitals')
plt.show()
hist, bins = np.histogram(merged_df['age'].dropna(), bins=[0, 15, 35, 55, 70, 80])
most_common_age_bin_index = np.argmax(hist)
most_common_age_range = f"{bins[most_common_age_bin_index]}-{bins[most_common_age_bin_index + 1]}"

# Second plot
diagnosis_count = merged_df['diagnosis'].value_counts()
most_common_diagnosis = diagnosis_count.idxmax()
plt.pie(diagnosis_count, labels=diagnosis_count.index, autopct='%1.1f%%')
plt.title('Most Common Diagnosis')
plt.show()

# Third plot
sns.violinplot(x='hospital', y='height', data=merged_df)
plt.title('Height Distribution by Hospital')
plt.show()

explanation_for_height_distribution = "Prenatal hospital has very low values, sports one very high values"

# Printed answers
print(f"The answer to the 1st question: {most_common_age_range}")
print(f"The answer to the 2nd question: {most_common_diagnosis}")
print(f"The answer to the 3rd question: {explanation_for_height_distribution}")
