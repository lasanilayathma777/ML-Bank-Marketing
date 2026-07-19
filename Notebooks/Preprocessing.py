import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank-full.csv")
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "bank_processed.csv")

#Load the data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")

    df.columns = [column_name.strip('"') for column_name in df.columns]

    text_columns = df.select_dtypes(include="object").columns
    for col in text_columns:
        df[col] = df[col].str.strip('"')

    return df


#Fix the pdays = -1 special value (found in EDA step 5)
def handle_pdays(df):
    print("\nHandling pdays = -1\n")

    df["was_contacted_before"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].replace(-1, 0)

    print("Added column 'was_contacted_before'")
    print("Replaced -1 with 0 in 'pdays'")
    return df


#Turn text categories into numbers (one-hot encoding)
def encode_categorical(df):
    print("\nEncoding categorical columns\n")

    categorical_cols = ["job", "marital", "education", "default", "housing",
                         "loan", "contact", "month", "poutcome"]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Target column: yes- 1, no- 0
    df_encoded["y"] = (df_encoded["y"] == "yes").astype(int)

    print("Columns before encoding:", df.shape[1])
    print("Columns after encoding:", df_encoded.shape[1])
    return df_encoded

def save_processed_data(df_encoded):
    print("\nSaving processed dataset\n")

    df_encoded.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved processed dataset to: {PROCESSED_PATH}")
    print(f"Shape: {df_encoded.shape}")

#Split into train and test sets
def split_data(df_encoded):
    print("\nSplitting into train and test sets\n")

    X = df_encoded.drop(columns=["y"])
    y = df_encoded["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)
    print("Train class balance:\n", y_train.value_counts(normalize=True).round(3))
    return X_train, X_test, y_train, y_test


#Scale the numeric columns
def scale_numeric(X_train, X_test):
    print("\nScaling numeric columns\n")

    numeric_cols = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

    scaler = StandardScaler()

    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    print("Scaled columns:", numeric_cols)
    return X_train, X_test


def main():
    df = load_data()
    df = handle_pdays(df)
    df_encoded = encode_categorical(df)
    save_processed_data(df_encoded)
    X_train, X_test, y_train, y_test = split_data(df_encoded)
    X_train, X_test = scale_numeric(X_train, X_test)

    print("\nPreprocessing complete.")
    print("Final number of features:", X_train.shape[1])

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
