import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank-full.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "ModelComparison_outputs")


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


#Retrain both tuned models using the best parameters already found
def train_final_models(X_train, y_train):
    print("\nTraining final Decision Tree (best parameters from earlier tuning)\n")
    dt_model = DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, min_samples_split=2,
        criterion="gini", class_weight="balanced", random_state=42
    )
    dt_model.fit(X_train, y_train)

    print("Training final Neural Network (best parameters from earlier tuning)\n")
    nn_model = MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=0.01, learning_rate_init=0.01,
        activation="relu", solver="adam", max_iter=300,
        random_state=42, early_stopping=True
    )
    nn_model.fit(X_train, y_train)

    return dt_model, nn_model


#metrics table for both models
def print_comparison_table(y_test, dt_pred, dt_proba, nn_pred, nn_proba):
    print("\nModel comparison table\n")

    results = pd.DataFrame({
        "Model": ["Decision Tree", "Neural Network"],
        "Accuracy": [accuracy_score(y_test, dt_pred), accuracy_score(y_test, nn_pred)],
        "Precision": [precision_score(y_test, dt_pred), precision_score(y_test, nn_pred)],
        "Recall": [recall_score(y_test, dt_pred), recall_score(y_test, nn_pred)],
        "F1-Score": [f1_score(y_test, dt_pred), f1_score(y_test, nn_pred)],
        "ROC-AUC": [roc_auc_score(y_test, dt_proba), roc_auc_score(y_test, nn_proba)],
    })
    print(results.round(3).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results.round(3).to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)
    print("\nSaved table:", os.path.join(OUTPUT_DIR, "comparison_table.csv"))
    return results


#ROC curves for both models on the same plot
def plot_roc_curves(y_test, dt_proba, nn_proba):
    print("\nPlotting ROC curves\n")

    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_proba)
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(dt_fpr, dt_tpr, color="blue",
             label=f"Decision Tree (AUC = {roc_auc_score(y_test, dt_proba):.3f})")
    plt.plot(nn_fpr, nn_tpr, color="green",
             label=f"Neural Network (AUC = {roc_auc_score(y_test, nn_proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Decision Tree vs Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "roc_curves.png"))


#Precision-Recall curves for both models on the same plot
def plot_pr_curves(y_test, dt_proba, nn_proba):
    print("\nPlotting precision-recall curves\n")

    dt_precision, dt_recall, _ = precision_recall_curve(y_test, dt_proba)
    nn_precision, nn_recall, _ = precision_recall_curve(y_test, nn_proba)

    plt.figure(figsize=(7, 6))
    plt.plot(dt_recall, dt_precision, color="blue", label="Decision Tree")
    plt.plot(nn_recall, nn_precision, color="green", label="Neural Network")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Decision Tree vs Neural Network")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curves.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "pr_curves.png"))


#confusion matrices
def plot_side_by_side_confusion(y_test, dt_pred, nn_pred):
    print("\nPlotting side-by-side confusion matrices\n")

    dt_cm = confusion_matrix(y_test, dt_pred)
    nn_cm = confusion_matrix(y_test, nn_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title, cmap in [
        (axes[0], dt_cm, "Decision Tree", "Blues"),
        (axes[1], nn_cm, "Neural Network", "Greens"),
    ]:
        im = ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No", "Yes"])
        ax.set_yticklabels(["No", "Yes"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(f"{title} - Confusion Matrix")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "side_by_side_confusion.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "side_by_side_confusion.png"))


#Bar chart comparing all metrics side by side
def plot_metrics_bar_chart(results):
    print("\nPlotting metrics bar chart\n")

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.35

    dt_values = results.loc[results["Model"] == "Decision Tree", metrics].values.flatten()
    nn_values = results.loc[results["Model"] == "Neural Network", metrics].values.flatten()

    plt.figure(figsize=(9, 6))
    plt.bar(x - width/2, dt_values, width, label="Decision Tree", color="steelblue")
    plt.bar(x + width/2, nn_values, width, label="Neural Network", color="mediumseagreen")
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Performance Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "metrics_comparison.png"))


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess()

    dt_model, nn_model = train_final_models(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)[:, 1]

    nn_pred = nn_model.predict(X_test)
    nn_proba = nn_model.predict_proba(X_test)[:, 1]

    results = print_comparison_table(y_test, dt_pred, dt_proba, nn_pred, nn_proba)
    plot_roc_curves(y_test, dt_proba, nn_proba)
    plot_pr_curves(y_test, dt_proba, nn_proba)
    plot_side_by_side_confusion(y_test, dt_pred, nn_pred)
    plot_metrics_bar_chart(results)

    print("\nModel comparison complete.")


if __name__ == "__main__":
    main()
