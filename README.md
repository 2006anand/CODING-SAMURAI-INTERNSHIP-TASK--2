import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

df = sns.load_dataset('iris')

print("Dataset Shape:", df.shape)
print(df.head())

plt.figure(figsize=(8,6))
sns.pairplot(df, hue='species', corner=True)
plt.show()

corr = df.drop('species', axis=1).corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True)
plt.show()

X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Accuracy:", np.round(scores.mean(), 4))

joblib.dump(model, "iris_random_forest.joblib")
print("Model saved as iris_random_forest.joblib")

sample_df = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2],
     [6.7, 3.0, 5.2, 2.3]],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
)

preds = model.predict(sample_df)
print("Sample Predictions:", preds)

loaded = joblib.load("iris_random_forest.joblib")
print("Loaded Model Prediction:", loaded.predict(X_test.iloc[:5]))
