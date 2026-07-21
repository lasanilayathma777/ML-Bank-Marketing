import os
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank-full.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "EDA_outputs")


#Load the data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df.columns = [column_name.strip('"') for column_name in df.columns]

    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].str.strip('"')

    return df


#Basic checks - shape, types, missing values
def basic_checks(df):
    print("\nBasic checks\n")

    print("Number of rows and columns:", df.shape)
    print("\nData type of each column:")
    print(df.dtypes)

    print("\nMissing (blank/NaN) values in each column:")
    print(df.isnull().sum())

#Look at the target column (y) - is it balanced?
def target_distribution(df):
    print("\nTarget class distribution (y)\n")

    counts = df["y"].value_counts()
    percentages = df["y"].value_counts(normalize=True)

    print("Counts:\n", counts)
    print("\nPercentages:\n", percentages.round(3))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    counts.plot(kind="bar", color=["steelblue", "orange"])
    plt.title("How many clients said yes/no to a term deposit")
    plt.ylabel("Number of clients")
    plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"))
    plt.close()
    print("\nSaved chart: EDA_outputs/target_distribution.png")


#Check for hidden "unknown" values in text columns
def check_unknown_values(df):
    print("\n'Unknown' value check\n")
    print("'unknown' is not a blank/NaN, but it means the same thing:")
    print("the bank didn't record this information for that client.\n")

    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        count = (df[col] == "unknown").sum()
        if count > 0:
            percent = count / len(df) * 100
            print(f"  {col}: {count} unknowns ({percent:.1f}% of all rows)")


#Check the special pdays = -1 value
def check_pdays(df):
    print("\npdays = -1 check\n")
    print("pdays = 'days since last contact'. But -1 is a special code")
    print("meaning 'this client was never contacted before' - not a real")
    print("day count. We need to know how common this is.\n")

    count = (df["pdays"] == -1).sum()
    percent = count / len(df) * 100
    print(f"Clients never contacted before: {count} ({percent:.1f}%)")


#Correlation between numeric features and the target
def correlation_matrix(df):
    print("\nCorrelation matrix\n")

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

    # Make a temporary numeric copy of the target so it can be included
    # in the correlation calculation (correlation only works on numbers)
    df_corr = df[numeric_cols].copy()
    df_corr["y"] = (df["y"] == "yes").astype(int)

    corr = df_corr.corr()
    print(corr["y"].sort_values())

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    # Print the correlation number inside each cell
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Correlation Matrix (numeric features + target)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
    plt.close()
    print("\nSaved chart:", os.path.join(OUTPUT_DIR, "correlation_matrix.png"))

    print("\nNote: 'duration' has the strongest correlation with y (~0.39),")
    print("but duration is only known AFTER a call happens - a real-world")
    print("limitation worth discussing in the report.")


#Boxplots - show spread and outliers in numeric features
def boxplots(df):
    print("\nBoxplots\n")

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 5))
    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col])
        ax.set_title(col)
    plt.suptitle("Boxplots of Numeric Features - Distribution and Outliers")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "boxplots.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "boxplots.png"))

#Histograms - show the shape of each numeric feature's distribution
def histograms(df):
    print("\nHistograms\n")

    numeric_cols = ["age", "balance", "duration", "campaign", "pdays", "previous"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=30, color="salmon", edgecolor="black")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
    plt.suptitle("Histograms of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "histograms.png"))


#Bar plots - distribution of key categorical features
def categorical_bar_plots(df):
    print("\nCategorical bar plots\n")

    categorical_cols = ["job", "marital", "education", "default",
                         "housing", "loan", "contact", "poutcome"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, categorical_cols):
        counts = df[col].value_counts()
        ax.bar(counts.index, counts.values, color="skyblue", edgecolor="black")
        ax.set_title(f"{col} Distribution")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle("Bar Plots: Categorical Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categorical_bar_plots.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "categorical_bar_plots.png"))


#Count plots - categorical features compared against the target (y)
def categorical_count_plots(df):
    print("\nCategorical count plots vs target\n")

    categorical_cols = ["job", "marital", "education", "default",
                         "housing", "loan", "contact", "poutcome"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()
    for ax, col in zip(axes, categorical_cols):
        cross = pd.crosstab(df[col], df["y"])
        cross.plot(kind="bar", ax=ax, color=["lightblue", "salmon"])
        ax.set_title(f"{col} vs y")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="y")
    plt.suptitle("Count Plots: Categorical Features with Target Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "categorical_count_plots.png"))
    plt.close()
    print("Saved chart:", os.path.join(OUTPUT_DIR, "categorical_count_plots.png"))

def main():
    df = load_data()
    basic_checks(df)
    target_distribution(df)
    check_unknown_values(df)
    check_pdays(df)
    correlation_matrix(df)
    boxplots(df)
    histograms(df)
    categorical_bar_plots(df)
    categorical_count_plots(df)

if __name__ == "__main__":
    main()
