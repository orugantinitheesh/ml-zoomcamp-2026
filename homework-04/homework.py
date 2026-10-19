
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "course_lead_scoring_2026.csv")

target = "converted"

# Split off test data, then create train and validation sets
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1
)

df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Separate target
y_train = df_train[target]
y_val = df_val[target]


def prepare_features(data):
    data = data.drop(columns=[target]).copy()

    categorical = data.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    numerical = data.select_dtypes(
        include=["number"]
    ).columns

    data[categorical] = data[categorical].fillna("NA")
    data[numerical] = data[numerical].fillna(0)

    return data.to_dict(orient="records")


def train_model(train_data, validation_data, C=1.0):
    train_dicts = prepare_features(train_data)
    val_dicts = prepare_features(validation_data)

    dv = DictVectorizer(sparse=True)

    X_train = dv.fit_transform(train_dicts).astype(np.float64)
    X_val = dv.transform(val_dicts).astype(np.float64)

    X_train.indices = X_train.indices.astype(np.int32)
    X_train.indptr = X_train.indptr.astype(np.int32)

    X_val.indices = X_val.indices.astype(np.int32)
    X_val.indptr = X_val.indptr.astype(np.int32)

    model = LogisticRegression(
        solver="liblinear",
        C=C,
        max_iter=1000
    )

    model.fit(X_train, train_data[target])

    probabilities = model.predict_proba(X_val)[:, 1]

    return model, dv, X_train, X_val, probabilities



print("\nQUESTION 1: ROC AUC FEATURE IMPORTANCE")

features = [
    "lead_score",
    "number_of_courses_viewed",
    "interaction_count",
    "annual_income"
]

for feature in features:
    scores = df_train[feature].fillna(0)
    auc = roc_auc_score(y_train, scores)

    if auc < 0.5:
        auc = roc_auc_score(y_train, -scores)

    print(f"{feature}: {auc:.4f}")


print("\nQUESTION 2: MODEL AUC")

model, dv, X_train, X_val, y_prob = train_model(
    df_train, df_val, C=1.0
)

auc = roc_auc_score(y_val, y_prob)
print(f"Validation AUC: {auc:.3f}")


print("\nQUESTION 3: PRECISION AND RECALL")

thresholds = np.arange(0.0, 1.001, 0.01)

precision_values = []
recall_values = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    precision_values.append(
        precision_score(y_val, y_pred, zero_division=0)
    )
    recall_values.append(
        recall_score(y_val, y_pred, zero_division=0)
    )

precision_values = np.array(precision_values)
recall_values = np.array(recall_values)

valid = ~(
    (precision_values == 0) & (recall_values == 0)
)

differences = np.abs(precision_values - recall_values)
differences[~valid] = np.inf

best_intersection = np.argmin(differences)

print(f"Intersection threshold: {thresholds[best_intersection]:.2f}")
print(f"Precision: {precision_values[best_intersection]:.4f}")
print(f"Recall: {recall_values[best_intersection]:.4f}")

plt.plot(thresholds, precision_values, label="Precision")
plt.plot(thresholds, recall_values, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision and Recall vs Threshold")
plt.legend()
plt.grid(True)


print("\nQUESTION 4: F1 SCORE")

f1_values = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    p = precision_score(y_val, y_pred, zero_division=0)
    r = recall_score(y_val, y_pred, zero_division=0)

    if p + r == 0:
        f1 = 0
    else:
        f1 = 2 * p * r / (p + r)

    f1_values.append(f1)

f1_values = np.array(f1_values)
best_f1_index = np.argmax(f1_values)

print(f"Best threshold: {thresholds[best_f1_index]:.2f}")
print(f"Maximum F1 score: {f1_values[best_f1_index]:.4f}")


print("\nQUESTION 5: 5-FOLD CROSS-VALIDATION")

kf = KFold(n_splits=5, shuffle=True, random_state=1)
cv_scores = []

for train_idx, val_idx in kf.split(df_full_train):
    fold_train = df_full_train.iloc[train_idx]
    fold_val = df_full_train.iloc[val_idx]

    _, _, _, _, fold_prob = train_model(
        fold_train, fold_val, C=1.0
    )

    fold_auc = roc_auc_score(
        fold_val[target], fold_prob
    )

    cv_scores.append(fold_auc)

cv_scores = np.array(cv_scores)

print("Fold AUC scores:", np.round(cv_scores, 4))
print(f"Mean AUC: {cv_scores.mean():.3f}")
print(f"Standard deviation: {cv_scores.std():.3f}")


print("\nQUESTION 6: HYPERPARAMETER TUNING")

C_values = [0.000001, 0.001, 1]
results = []

for C in C_values:
    scores = []

    for train_idx, val_idx in kf.split(df_full_train):
        fold_train = df_full_train.iloc[train_idx]
        fold_val = df_full_train.iloc[val_idx]

        _, _, _, _, fold_prob = train_model(
            fold_train, fold_val, C=C
        )

        fold_auc = roc_auc_score(
            fold_val[target], fold_prob
        )

        scores.append(fold_auc)

    scores = np.array(scores)
    mean_auc = scores.mean()
    std_auc = scores.std()

    results.append((C, mean_auc, std_auc))

    print(
        f"C={C}: Mean AUC={mean_auc:.3f}, "
        f"Std={std_auc:.3f}"
    )

best_result = max(results, key=lambda result: result[1])

print(f"\nBest C: {best_result[0]}")
print(f"Best mean AUC: {best_result[1]:.3f}")
print(f"Standard deviation: {best_result[2]:.3f}")
plt.show()