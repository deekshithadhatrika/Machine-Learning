import numpy as np

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("EXPERIMENT 5: FULL LOGISTIC REGRESSION PIPELINE")
print("=" * 60)

# ----------------------------------------------------------
# 1. BINARY LOGISTIC REGRESSION
# ----------------------------------------------------------

print("\nBINARY LOGISTIC REGRESSION")
print("-" * 40)

data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ----------------------------------------------------------
# 2. REGULARIZATION
# ----------------------------------------------------------

print("\nREGULARIZATION COMPARISON")
print("-" * 40)

models = {
    "L1 Regularization": LogisticRegression(
        penalty="l1",
        solver="liblinear",
        max_iter=5000
    ),

    "L2 Regularization": LogisticRegression(
        penalty="l2",
        max_iter=5000
    ),

    "Elastic Net": LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000
    )
}

for name, logistic_model in models.items():

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", logistic_model)
    ])

    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print(name, ":", round(accuracy, 4))

# ----------------------------------------------------------
# 3. BIAS-VARIANCE TRADE-OFF
# ----------------------------------------------------------

print("\nBIAS-VARIANCE TRADE-OFF")
print("-" * 40)

C_values = [0.01, 0.1, 1, 10, 100]

for c in C_values:

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            C=c,
            max_iter=5000
        ))
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    print(
        "C =", c,
        "| Mean Accuracy =",
        round(scores.mean(), 4)
    )

# ----------------------------------------------------------
# 4. MULTINOMIAL LOGISTIC REGRESSION
# ----------------------------------------------------------

print("\nMULTINOMIAL LOGISTIC REGRESSION")
print("-" * 40)

wine = load_wine()

X_wine = wine.data
y_wine = wine.target

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_wine,
    y_wine,
    test_size=0.2,
    random_state=42,
    stratify=y_wine
)

multinomial_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=5000
    ))
])

multinomial_model.fit(X_train_w, y_train_w)

wine_pred = multinomial_model.predict(X_test_w)

print(
    "Multinomial Accuracy:",
    round(accuracy_score(y_test_w, wine_pred), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_w, wine_pred))

print("\nClassification Report:")
print(classification_report(y_test_w, wine_pred))

print("=" * 60)
print("EXPERIMENT 5 COMPLETED SUCCESSFULLY")
print("=" * 60)