import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split


df = pd.read_csv("data3.csv")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


print("\nQ1: Mode for industry")
mode_industry = df["industry"].mode()[0]
print("Answer:", mode_industry)


target_column = "converted"

X = df.drop(columns=[target_column]).copy()
y = df[target_column]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numerical = X.select_dtypes(exclude=["object"]).columns.tolist()

X[categorical] = X[categorical].fillna("NA")
X[numerical] = X[numerical].fillna(0.0)

data = X.copy()
data[target_column] = y

train_full, test = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

train, validation = train_test_split(
    train_full,
    test_size=0.25,
    random_state=42
)

y_train = train.pop(target_column)
y_validation = validation.pop(target_column)

print("\nTraining rows:", len(train))
print("Validation rows:", len(validation))
print("Test rows:", len(test))

print("\nQ2: Biggest correlation")

correlations = X[numerical].corr().abs()

candidate_pairs = [
    ("interaction_count", "lead_score"),
    ("number_of_courses_viewed", "lead_score"),
    ("number_of_courses_viewed", "interaction_count"),
    ("annual_income", "interaction_count")
]

best_pair = max(
    candidate_pairs,
    key=lambda pair: correlations.loc[pair[0], pair[1]]
)

print("Answer:", best_pair)
print(
    "Correlation:",
    correlations.loc[best_pair[0], best_pair[1]]
)

print("\nQ3: Biggest Mutual Information")

mi_scores = {}

for column in categorical:
    mi_scores[column] = mutual_info_score(
        y_train,
        train[column]
    )

for column, score in sorted(
    mi_scores.items(),
    key=lambda item: item[1],
    reverse=True
):
    print(f"{column}: {score:.6f}")

best_mi_feature = max(mi_scores, key=mi_scores.get)

print("Answer:", best_mi_feature)

print("\nQ4: Accuracy")

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(
    train.to_dict(orient="records")
)

X_validation = dv.transform(
    validation.to_dict(orient="records")
)

model = LogisticRegression(
    solver="liblinear",
    C=1.0,
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_validation)
accuracy = accuracy_score(y_validation, predictions)
print("Accuracy:", round(accuracy, 2))
print("\nQ5: Feature Selection")

original_accuracy = accuracy_score(y_validation, predictions)

print("Original accuracy:", original_accuracy)

features_to_test = [
    "lead_source",
    "number_of_courses_viewed",
    "interaction_count"
]

feature_differences = {}

for feature in features_to_test:

    train_reduced = train.drop(columns=[feature])
    validation_reduced = validation.drop(columns=[feature])

    dv_reduced = DictVectorizer(sparse=False)

    X_train_reduced = dv_reduced.fit_transform(
        train_reduced.to_dict(orient="records")
    )

    X_val_reduced = dv_reduced.transform(
        validation_reduced.to_dict(orient="records")
    )

    model_reduced = LogisticRegression(
        solver="liblinear",
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    model_reduced.fit(X_train_reduced, y_train)

    predictions_reduced = model_reduced.predict(X_val_reduced)

    accuracy_reduced = accuracy_score(
        y_validation,
        predictions_reduced
    )

    difference = original_accuracy - accuracy_reduced

    feature_differences[feature] = difference

    print(
        f"{feature}: accuracy={accuracy_reduced:.6f}, "
        f"difference={difference:.6f}"
    )

best_feature = min(feature_differences, key=feature_differences.get)

print("Q5 Answer:", best_feature)

print("\nQ6: Parameter Tuning")

C_values = [0.000001, 0.00001, 0.0001, 0.001]

accuracies = {}

for C_value in C_values:

    model_tuned = LogisticRegression(
        solver="liblinear",
        C=C_value,
        max_iter=1000,
        random_state=42
    )

    model_tuned.fit(X_train, y_train)

    predictions_tuned = model_tuned.predict(X_validation)

    accuracy_tuned = accuracy_score(
        y_validation,
        predictions_tuned
    )

    accuracies[C_value] = accuracy_tuned

    print(
        f"C={C_value}: accuracy={accuracy_tuned:.3f}"
    )

best_C = max(C_values, key=lambda c: (accuracies[c], -c))

print("Q6 Answer:", best_C)