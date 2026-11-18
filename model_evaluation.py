
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

# Load original dataset
df = pd.read_csv("Titanic.csv")

# Feature engineering
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Features
X = df[[
    'sex', 'age', 'sibsp', 'parch',
    'fare', 'embarked', 'class',
    'family_size', 'is_alone'
]]

y = df['survived']

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained Logistic Regression pipeline
model = joblib.load("Logistic_Regression.pkl")

# Predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Did Not Survive", "Survived"],
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ROC-AUC
auc = roc_auc_score(y_test, probabilities)

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"Logistic Regression AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("ROC-AUC Score:", round(auc, 4))


