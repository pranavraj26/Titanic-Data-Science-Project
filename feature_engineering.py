
import pandas as pd
import numpy as np

# Load original Titanic dataset
df = pd.read_csv("Titanic.csv")

print("Dataset Loaded Successfully!")
print("Original Shape:", df.shape)


# --------------------------------
# FEATURE 1: FAMILY SIZE
# --------------------------------

df['family_size'] = (
    df['sibsp'] + df['parch'] + 1
)

print("\nFamily Size Distribution:")
print(df['family_size'].value_counts().sort_index())


# --------------------------------
# FEATURE 2: IS ALONE
# --------------------------------

df['is_alone'] = (
    df['family_size'] == 1
).astype(int)

print("\nPassengers Travelling Alone:")
print(df['is_alone'].value_counts())


# --------------------------------
# FEATURE ANALYSIS
# --------------------------------

family_analysis = df.groupby(
    'family_size'
)['survived'].mean() * 100

print("\nFamily Size vs Survival Rate:")
print(family_analysis)


# --------------------------------
# PREPARE X AND Y
# --------------------------------

X = df[[
    'sex',
    'age',
    'sibsp',
    'parch',
    'fare',
    'embarked',
    'class',
    'family_size',
    'is_alone'
]]

y = df['survived']


# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- DATA SPLITTING ---")

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
