import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank-full.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "DecisionTree_outputs")


#Load and preprocess data
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = [c.strip('"') for c in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip('"')

    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].replace(-1, 0)

    categorical_cols = ["job", "marital", "education", "default", "housing",
                         "loan", "contact", "month", "poutcome"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded["y"] = (df_encoded["y"] == "yes").astype(int)

    X = df_encoded.drop(columns=["y"])
    y = df_encoded["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    return X_train, X_test, y_train, y_test

def print_metrics(y_true, y_pred, label):
    print(f"\n{label}")
    print("Accuracy: ", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred), 3))
    print("Recall:   ", round(recall_score(y_true, y_pred), 3))
    print("F1 score: ", round(f1_score(y_true, y_pred), 3))


#Baseline tree
def train_baseline(X_train, y_train):
    print("\nTraining baseline Decision Tree (default parameters)\n")

    model = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


#Tuned tree: GridSearchCV
def train_tuned(X_train, y_train):
    print("\nTuning Decision Tree with GridSearchCV\n")

    param_grid = {
        "max_depth": [4, 6, 8, 10, 12, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    base_model = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    grid = GridSearchCV(
        base_model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)
    print("Best cross-validation F1 score:", round(grid.best_score_, 3))
    return grid.best_estimator_


#Compare baseline vs tuned performance with a bar chart
def compare_baseline_vs_tuned(y_test, y_pred_baseline, y_pred_tuned):
    print("\nComparing baseline vs tuned performance\n")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

    baseline_scores = [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_baseline),
    ]
    tuned_scores = [
        accuracy_score(y_test, y_pred_tuned),
        precision_score(y_test, y_pred_tuned),
        recall_score(y_test, y_pred_tuned),
        f1_score(y_test, y_pred_tuned),
    ]

    comparison = pd.DataFrame({
        "Metric": metrics,
        "Baseline": baseline_scores,
        "Tuned": tuned_scores,
    })
    comparison["Improvement %"] = (
        (comparison["Tuned"] - comparison["Baseline"]) / comparison["Baseline"] * 100
    )
    print(comparison.round(3).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    x = np.arange(len(metrics))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x - width/2, baseline_scores, width, label="Baseline", color="skyblue")
    axes[0].bar(x + width/2, tuned_scores, width, label="Tuned", color="salmon")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_title("Decision Tree: Baseline vs Tuned Performance")
    axes[0].legend()

    colors = ["green" if v > 0 else "red" for v in comparison["Improvement %"]]
    axes[1].bar(metrics, comparison["Improvement %"], color=colors)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title("Percentage Improvement After Tuning")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_vs_tuned.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "baseline_vs_tuned.png"))


#Feature importance
def feature_importance(model, X_train):
    print("\nFeature importance (top 10)\n")

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print(top10.round(3))

    plt.figure(figsize=(8, 6))
    top10.sort_values().plot(kind="barh", color="teal")
    plt.title("Decision Tree - Top 10 Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "feature_importance.png"))


#Learning curve
def plot_learning_curve(model, X_train, y_train):
    print("\nGenerating learning curve (this may take a moment)\n")

    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", color="red", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="green", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "learning_curve.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "learning_curve.png"))
    print(f"Final training score: {train_mean[-1]:.3f}")
    print(f"Final CV score: {test_mean[-1]:.3f}")
    print(f"Gap (overfitting indicator): {train_mean[-1] - test_mean[-1]:.3f}")


#Cross-validation
def cross_validation(model, X_train, y_train):
    print("\n5-fold cross-validation\n")

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1)

    print("Fold accuracies:", scores.round(3))
    print(f"Mean accuracy: {scores.mean():.3f}")
    print(f"Standard deviation: {scores.std():.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 6), scores, "bo-", linewidth=2, markersize=8)
    plt.axhline(y=scores.mean(), color="red", linestyle="--", label=f"Mean: {scores.mean():.3f}")
    plt.xlabel("Fold Number")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree Cross-Validation Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "cross_validation.png"))


#Confusion matrix for the final tuned model
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print("             predicted no  predicted yes")
    print("actual no   ", cm[0])
    print("actual yes  ", cm[1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Decision Tree - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    # Baseline
    baseline_model = train_baseline(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    print_metrics(y_test, y_pred_baseline, "Baseline results on test set")

    # Tuned
    tuned_model = train_tuned(X_train, y_train)
    y_pred_tuned = tuned_model.predict(X_test)
    print_metrics(y_test, y_pred_tuned, "Tuned results on test set")

    # Comparison chart
    compare_baseline_vs_tuned(y_test, y_pred_baseline, y_pred_tuned)

    # Extra analysis on the tuned (final) model
    feature_importance(tuned_model, X_train)
    plot_confusion_matrix(y_test, y_pred_tuned)
    plot_learning_curve(tuned_model, X_train, y_train)
    cross_validation(tuned_model, X_train, y_train)

    print("\nDecision Tree analysis complete.")


if __name__ == "__main__":
    main()
