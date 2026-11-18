import pandas as pd
import numpy as np

# Load Titanic dataset
df = pd.read_csv(r"C:\Users\pranav\Downloads\Titanic.csv")

# Display first 5 rows
print(df.head())

print("\n--- DATASET DIMENSIONS ---")

print("Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])
print("Total Values:", df.size)
print("\n--- COLUMN NAMES ---")

print(df.columns.tolist())

print("\n--- DATASET INFORMATION ---")

df.info()

print("\n--- STATISTICAL SUMMARY ---")

print(df.describe())

print("\n--- MISSING VALUES ---")

print(df.isnull().sum())

print("\n--- DUPLICATE RECORDS ---")

print("Total duplicates:", df.duplicated().sum())

missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Value Percentage:")
print(missing_percentage)

duplicates = df[df.duplicated(keep=False)]

print("\nDuplicate Records:")
print(duplicates.head(20))

print("\nTotal Duplicate Rows:", len(duplicates))

missing_age = df[df['age'].isnull()]

print("\nPassengers with Missing Age:")
print(missing_age.head(10))

print("Total Missing Ages:", len(missing_age))

missing_embarked = df[df['embarked'].isnull()]

print("\nMissing Embarked Records:")
print(missing_embarked)

Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df['fare'] < lower_limit) |
    (df['fare'] > upper_limit)
]

print("\nFare Outliers:")
print(outliers[['fare', 'class', 'survived']])

print("Total Outliers:", len(outliers))

# Create a copy of original dataset

df_clean = df.copy()

print("Original dataset:", df.shape)
print("Working dataset:", df_clean.shape)

print("Mean Age:", df_clean['age'].mean())

print("Median Age:", df_clean['age'].median())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df_clean,
    x='age',
    bins=30,
    kde=True
)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Frequency")

plt.show()

age_median = df_clean['age'].median()

print("Age Median:", age_median)

df_clean['age'] = df_clean['age'].fillna(age_median)

print("Missing Ages After Cleaning:")
print(df_clean['age'].isnull().sum())

df_clean = df.copy()

group_medians = df_clean.groupby(
    ['sex', 'class'],
    observed=True
)['age'].transform('median')

df_clean['age'] = df_clean['age'].fillna(group_medians)

# Fallback if any age is still missing
df_clean['age'] = df_clean['age'].fillna(
    df_clean['age'].median()
)

print("Remaining Missing Ages:",
    df_clean['age'].isnull().sum())

print(df_clean['embarked'].value_counts())

embarked_mode = df_clean['embarked'].mode()[0]

print("Most Common Embarked:", embarked_mode)

df_clean['embarked'] = df_clean['embarked'].fillna(
    embarked_mode
)

print("Remaining Missing Embarked:",
    df_clean['embarked'].isnull().sum())

duplicates = df_clean[
    df_clean.duplicated(keep=False)
]

print("Duplicate Records:")
print(duplicates.head(10))

print("Total Duplicate Rows:",
    df_clean.duplicated().sum())

Q1 = df_clean['fare'].quantile(0.25)
Q3 = df_clean['fare'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df_clean[
    (df_clean['fare'] < lower_limit) |
    (df_clean['fare'] > upper_limit)
]

print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)

print("Total Fare Outliers:", len(outliers))
plt.figure(figsize=(10, 5))

sns.boxplot(data=df_clean, x='fare')

plt.title("Fare Outlier Detection")

plt.show()
print("\n--- FINAL DATA QUALITY REPORT ---")

print("Original Shape:", df.shape)
print("Cleaned Shape:", df_clean.shape)

print("\nMissing Values:")
print(df_clean.isnull().sum())

print("\nDuplicate Records:")
print(df_clean.duplicated().sum())

print("\nData Types:")
print(df_clean.dtypes)

df_clean.to_csv(
    "Titanic_cleaned.csv",
    index=False
)

print("Cleaned dataset exported successfully!")

# ==================================
# EXPLORATORY DATA ANALYSIS
# ==================================

print("\n--- SURVIVAL ANALYSIS ---")

print(df_clean['survived'].value_counts())

survival_rate = df_clean['survived'].mean() * 100

print("Overall Survival Rate:", round(survival_rate, 2), "%")

plt.figure(figsize=(8, 5))

sns.countplot(data=df_clean, x='survived')

plt.title("Titanic Passenger Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.show()

gender_survival = df_clean.groupby(
    'sex'
)['survived'].mean() * 100

print(gender_survival)

gender_survival.plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title("Survival Rate by Gender")
plt.ylabel("Survival Percentage")
plt.xlabel("Gender")
plt.xticks(rotation=0)

plt.show()

class_survival = df_clean.groupby(
    'class',
    observed=True
)['survived'].mean() * 100

print(class_survival)

class_survival.plot(
    kind='bar',
    figsize=(8, 5)
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Percentage")
plt.xticks(rotation=0)

plt.show()

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df_clean,
    x='age',
    hue='survived',
    bins=25,
    kde=True
)

plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Passenger Count")

plt.show()

df_clean['age_group'] = pd.cut(
    df_clean['age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teenager', 'Young Adult',
            'Adult', 'Senior'],
    include_lowest=True
)

age_survival = df_clean.groupby(
    'age_group',
    observed=True
)['survived'].mean() * 100

print("\nSurvival Rate by Age Group:")
print(age_survival)
df_clean['family_size'] = (
    df_clean['sibsp'] +
    df_clean['parch'] + 1
)

print("\nFamily Size Distribution:")
print(df_clean['family_size'].value_counts().sort_index())
family_survival = df_clean.groupby(
    'family_size'
)['survived'].mean() * 100

print("\nFamily Size vs Survival:")
print(family_survival)
embarked_survival = df_clean.groupby(
    'embarked'
)['survived'].mean() * 100

print("\nSurvival by Embarked:")
print(embarked_survival)

plt.figure(figsize=(10, 7))

numeric_df = df_clean.select_dtypes(
    include=['number']
)

correlation = numeric_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title("Titanic Correlation Heatmap")
plt.show()