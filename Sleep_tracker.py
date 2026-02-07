import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# CREATE OUTPUT FOLDER

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# STEP 1: LOAD DATASET

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")


# STEP 2: SELECT REQUIRED COLUMNS

df = df[
    [
        "Age",
        "Sleep Duration",
        "Physical Activity Level",
        "Stress Level",
        "Quality of Sleep"
    ]
]

# STEP 3: LABEL SLEEP QUALITY

def label_sleep_quality(value):
    if value >= 8:
        return "Good"
    elif value >= 5:
        return "Average"
    else:
        return "Poor"

df["Sleep Quality Label"] = df["Quality of Sleep"].apply(label_sleep_quality)
df.drop("Quality of Sleep", axis=1, inplace=True)

# STEP 4: ENCODE TARGET

label_encoder = LabelEncoder()
df["Sleep Quality Label"] = label_encoder.fit_transform(df["Sleep Quality Label"])


# STEP 5: DATA VISUALIZATION (AUTO-SAVED)


# Sleep Quality Distribution
plt.figure()
sns.countplot(x="Sleep Quality Label", data=df)
plt.xticks(
    ticks=[0, 1, 2],
    labels=label_encoder.inverse_transform([0, 1, 2])
)
plt.title("Distribution of Sleep Quality")
plt.savefig(os.path.join(output_dir, "sleep_quality_distribution.png"))
plt.close()

# Stress vs Sleep Duration
plt.figure()
plt.scatter(
    df["Stress Level"],
    df["Sleep Duration"],
    c=df["Sleep Quality Label"],
    cmap="viridis"
)
plt.xlabel("Stress Level")
plt.ylabel("Sleep Duration (hours)")
plt.title("Stress Level vs Sleep Duration")
plt.savefig(os.path.join(output_dir, "stress_vs_sleep_duration.png"))
plt.close()

# STEP 6: TRAIN-TEST SPLIT

X = df.drop("Sleep Quality Label", axis=1)
y = df["Sleep Quality Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 7: FEATURE SCALING

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 8: TRAIN MODEL

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# STEP 9: MODEL EVALUATION
# -------------------------------
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# STEP 10: FEATURE IMPORTANCE (AUTO-SAVED)
# -------------------------------
plt.figure()
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance in Sleep Quality Prediction")
plt.savefig(os.path.join(output_dir, "feature_importance.png"))
plt.close()

# -------------------------------
# STEP 11: USER INPUT
# -------------------------------
print("\n--- Enter Your Details ---")

age = int(input("Age: "))
sleep_duration = float(input("Sleep Duration (hours): "))
activity = int(input("Physical Activity (minutes): "))
stress = int(input("Stress Level (0–10): "))

user_data = np.array([[age, sleep_duration, activity, stress]])
user_scaled = scaler.transform(user_data)

prediction = model.predict(user_scaled)
predicted_label = label_encoder.inverse_transform(prediction)[0]

print("\nPredicted Sleep Quality:", predicted_label)

# STEP 12: END OUTPUT VISUALIZATION


# Prediction bar
plt.figure()
plt.bar(
    ["Poor", "Average", "Good"],
    [
        1 if predicted_label == "Poor" else 0,
        1 if predicted_label == "Average" else 0,
        1 if predicted_label == "Good" else 0
    ]
)
plt.title("Your Predicted Sleep Quality")
plt.ylabel("Prediction")
plt.savefig(os.path.join(output_dir, "your_sleep_quality.png"))
plt.close()

# User vs Dataset Comparison
dataset_avg = X.mean()

plt.figure()
plt.bar(dataset_avg.index, dataset_avg.values, label="Dataset Average")
plt.bar(
    ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level"],
    user_data[0],
    alpha=0.6,
    label="Your Data"
)
plt.xticks(rotation=30)
plt.title("Your Data vs Dataset Average")
plt.legend()
plt.savefig(os.path.join(output_dir, "user_vs_dataset.png"))
plt.close()

# STEP 13: IMPROVEMENT-FOCUSED SUGGESTIONS
print("\nSuggestions:")

suggestion_given = False

if sleep_duration < 6:
    print("- Increase sleep duration to 7–8 hours for optimal recovery.")
    suggestion_given = True

if stress > 6:
    print("- Incorporate relaxation practices like meditation or journaling.")
    suggestion_given = True

if activity < 30:
    print("- Aim for at least 30–45 minutes of daily physical activity.")
    suggestion_given = True

if not suggestion_given:
    print("- Maintain a consistent sleep and wake schedule.")
    print("- Reduce screen exposure 30–60 minutes before bedtime.")
    print("- Optimize your sleep environment (dark, quiet, cool).")
    print("- Avoid heavy meals and caffeine late in the evening.")
    print("- Continue tracking sleep trends to sustain good sleep health.")

print("\nAll outputs and visualizations have been saved in the 'outputs' folder.")
