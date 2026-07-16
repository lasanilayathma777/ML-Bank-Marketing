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


def main():
    df = load_data()
    basic_checks(df)
    target_distribution(df)
    check_unknown_values(df)
    check_pdays(df)


if __name__ == "__main__":
    main()
