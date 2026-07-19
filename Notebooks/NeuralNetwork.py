import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank-full.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "NeuralNetwork_outputs")


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

#Baseline model
def train_baseline(X_train, y_train):
    print("\nTraining baseline Neural Network (single hidden layer)\n")

    model = MLPClassifier(
        hidden_layer_sizes=(16,),
        max_iter=300,
        random_state=42,
        early_stopping=True
    )
    model.fit(X_train, y_train)
    return model

#Tuned model
def train_tuned(X_train, y_train):
    print("\nTuning Neural Network with GridSearchCV\n")
    print("(This takes longer than the Decision Tree - MLPs are slower to fit)\n")

    param_grid = {
        "hidden_layer_sizes": [(16,), (32, 16), (64, 32)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }

    base_model = MLPClassifier(
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42,
        early_stopping=True
    )

    grid = GridSearchCV(
        base_model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print("Best parameters found:", grid.best_params_)
    print("Best cross-validation F1 score:", round(grid.best_score_, 3))
    return grid.best_estimator_


#Compare baseline vs tuned performance
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
    axes[0].bar(x + width/2, tuned_scores, width, label="Tuned", color="lightgreen")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_title("Neural Network: Baseline vs Tuned Performance")
    axes[0].legend()

    colors = ["green" if v > 0 else "red" for v in comparison["Improvement %"]]
    axes[1].bar(metrics, comparison["Improvement %"], color=colors)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_title("Percentage Improvement After Tuning")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_vs_tuned.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "baseline_vs_tuned.png"))


#Loss curve
def plot_loss_curve(model):
    print("\nPlotting training loss curve\n")

    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_, color="purple")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.title("Neural Network - Training Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "loss_curve.png"))
    print(f"Final training loss: {model.loss_curve_[-1]:.4f}")
    print(f"Number of iterations run: {model.n_iter_}")


#Confusion matrix for the final tuned model
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix:")
    print("             predicted no  predicted yes")
    print("actual no   ", cm[0])
    print("actual yes  ", cm[1])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Greens")
    plt.colorbar()
    plt.xticks([0, 1], ["No", "Yes"])
    plt.yticks([0, 1], ["No", "Yes"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Neural Network - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


#Cross-validation
def cross_validation(model, X_train, y_train):
    print("\n5-fold cross-validation\n")
    print("(Reduced fold count vs Decision Tree due to MLP training cost)\n")

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
    plt.title("Neural Network Cross-Validation Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cross_validation.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "cross_validation.png"))


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
    plot_loss_curve(tuned_model)
    plot_confusion_matrix(y_test, y_pred_tuned)
    cross_validation(tuned_model, X_train, y_train)

    print("\nNeural Network analysis complete.")


if __name__ == "__main__":
    main()
